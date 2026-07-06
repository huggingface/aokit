"""
AOKit Hub utilities

Helpers to compile a PyTorch module Ahead-of-Time and publish the resulting
packaged binary (plus samples, environment info, and an auto-generated README)
to a repository on the Hugging Face Hub. Designed to be driven from a
self-contained ``uv`` job script (see ``job.py`` at the repo root).
"""

import inspect
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable

import huggingface_hub as hf
import torch
import torch.utils.collect_env

from . import AOKIT_PACKAGE_ROOT
from . import load_from_package_dir
from . import load_source_code


# The default hardware flavor an AoT job targets: Nvidia RTX PRO 6000 (96 GB).
# Override at launch with `hf jobs uv run job.py --flavor <flavor>`.
# See https://huggingface.co/docs/hub/main/en/jobs-configuration#hardware-flavor
DEFAULT_FLAVOR = 'rtx-pro-6000'


def create_aoti_repo(
    module: torch.nn.Module,
    module_expr: str,
    compile_and_save: Callable[[torch.nn.Module, str], None],
    generate_samples: Callable[[str], None],
    aokit_loader: Callable[[torch.nn.Module, str], None] | None = None,
):
    """Compile a module Ahead-of-Time and publish it to a Hugging Face Hub repo.

    Orchestrates the full pipeline using the passed handlers:
    - generates output samples (before compilation)
    - compiles the module and saves the packaged binary
    - loads the compiled version (mutates the module)
    - generates output samples again (after compilation)
    - writes a README + environment/config context files
    - uploads everything to a freshly created Hub repo

    Parameters
    ----------
    module:
        Module to compile (e.g. `pipeline.transformer`).
    module_expr:
        String representation used in the generated README code.
    compile_and_save:
        `(module, package_dir) -> None`. Must AoT compile `module` (without
        mutating it) into `package_dir` using `aokit.compile_and_save`.
    generate_samples:
        `(samples_dir) -> None`. Must generate samples from the model and save
        them inside `samples_dir`.
    aokit_loader:
        Optional `(module, package_dir) -> None` custom loader passed to
        `aokit.load`. Defaults to `aokit.load_from_package_dir`.
    """

    HUB_URL = 'https://huggingface.co'

    # Default namespace comes from the authenticated user (whoami).
    user = hf.whoami()['name']
    job_id = os.environ.get('JOB_ID')
    job_info = _inspect_job(job_id)
    env_info = torch.utils.collect_env.get_env_info()
    library_name, config = _get_library_config(module)

    with TemporaryDirectory() as tempdir:
        tempdir = Path(tempdir)

        # Structure
        readme_path = tempdir / 'README.md'
        package_dir = tempdir / AOKIT_PACKAGE_ROOT  # `aokit.load` looks for this folder
        samples_before_dir = tempdir / 'samples' / 'before'
        samples_after_dir = tempdir / 'samples' / 'after'
        environment_path = tempdir / 'environment.json'
        config_path = tempdir / 'module_config.json'

        # Samples before compile
        samples_before_dir.mkdir(parents=True)
        t0 = time.perf_counter()
        generate_samples(str(samples_before_dir))
        generate_before_dt = time.perf_counter() - t0

        # Compile the packaged binary and load it back in
        package_dir.mkdir(parents=True)
        compile_and_save(module, str(package_dir))
        if aokit_loader is not None:
            aokit_loader(module, str(package_dir))
        else:
            load_from_package_dir(module, package_dir)

        # Samples after compile
        samples_after_dir.mkdir(parents=True)
        t0 = time.perf_counter()
        generate_samples(str(samples_after_dir))
        generate_after_dt = time.perf_counter() - t0

        # Environment and config dump
        environment_path.write_text(json.dumps(env_info._asdict(), indent=4))
        if config is not None:
            config_path.write_text(json.dumps(config, indent=4))

        # Create repo (namespace defaults to `user`)
        output_repo_id = _create_empty_repo(
            user=user,
            module=module,
            cuda_version=env_info.cuda_runtime_version,
        )

        # README.md
        model_init_region = (inspect.getsource(sys.modules['__main__'])
            .split('\n# README::MODEL_INIT::START')[1]
            .split('\n# README::MODEL_INIT::END')[0]
        )
        aokit_load_readme = load_source_code(
            module_expr=module_expr,
            repo_id=output_repo_id,
            aokit_loader=aokit_loader,
        )

        def get_link(path: Path):
            kind = 'tree' if path.is_dir() else 'resolve'
            return f'{HUB_URL}/{output_repo_id}/{kind}/main/{path.relative_to(tempdir)}'

        readme_path.write_text(_readme_template(
            model_init=model_init_region,
            aokit_load=aokit_load_readme,
            repo_id=output_repo_id,
            job_id=f'{user}/{job_id}' if job_id is not None else None,
            job_image=job_info.docker_image if job_info is not None else os.getenv('JOB_IMAGE'),
            job_flavor=job_info.flavor if job_info is not None else os.getenv('JOB_FLAVOR', DEFAULT_FLAVOR),
            environment=torch.utils.collect_env.pretty_str(env_info),
            library_name=library_name,
            generate_before_dt=generate_before_dt,
            generate_after_dt=generate_after_dt,
            samples_before_urls=[get_link(path) for path in samples_before_dir.iterdir()],
            samples_after_urls=[get_link(path) for path in samples_after_dir.iterdir()],
        ))

        # Self-include the entry-point job script so the repo is a reproducible
        # artifact of this Job. `__file__` here would point at this module, so
        # resolve the actual script being run via `__main__`.
        main_file = getattr(sys.modules['__main__'], '__file__', None)
        if main_file is not None:
            shutil.copyfile(main_file, tempdir / 'job.py')

        # Push everything (binary + samples + README + this script) to the Hub
        hf.upload_folder(repo_id=output_repo_id, folder_path=tempdir)
        print(f"AoT repository successfully created at: {HUB_URL}/{output_repo_id}")


def _inspect_job(job_id: str | None):
    """Best-effort lookup of the running Job's metadata (image, flavor)."""
    if job_id is None:
        return None
    inspect_job = getattr(hf, 'inspect_job', None)
    if inspect_job is None:  # older huggingface_hub
        return None
    try:
        return inspect_job(job_id=job_id)
    except Exception:
        return None


def _create_empty_repo(
    user: str,
    module: torch.nn.Module,
    cuda_version: str,
    max_attempts: int = 10,
):
    from requests.exceptions import HTTPError

    for _ in range(max_attempts):
        output_repo_id = _get_repo_id(user, module, cuda_version)
        try:
            hf.create_repo(output_repo_id)
        except HTTPError as err:
            if err.response.status_code != 409:  # 409 => name already taken, retry
                raise
        else:
            return output_repo_id
    raise AssertionError


def _get_repo_id(
    user: str,
    module: torch.nn.Module,
    cuda_version: str,
):
    # Full override, then namespace/base-name overrides, else derive from whoami.
    if (repo_id := os.getenv('OUTPUT_REPO_ID')) is not None:
        return repo_id
    namespace = os.getenv('OUTPUT_REPO_NAMESPACE', user)
    base_name = os.getenv('OUTPUT_REPO_BASE_NAME', module.__class__.__name__)
    sm = ''.join(map(str, torch.cuda.get_device_capability()))
    cu = ''.join(cuda_version.split('.')[:2]) if cuda_version else 'unknown'
    rnd = random.randbytes(1).hex()
    return f'{namespace}/{base_name}-sm{sm}-cu{cu}-r{rnd}'


def _get_library_config(module: torch.nn.Module):
    if (config := getattr(module, 'config', None)) is None:
        return None, None
    if callable(getattr(config, 'to_dict', None)):
        config = config.to_dict()
    if not isinstance(config, dict):
        return None, None
    if 'transformers_version' in config:
        library_name = 'transformers'
    elif '_diffusers_version' in config:
        library_name = 'diffusers'
    else:
        library_name = 'unknown'
    return library_name, config


def _readme_template(
    model_init: str,
    aokit_load: str,
    repo_id: str,
    job_id: str | None,
    job_image: str | None,
    job_flavor: str | None,
    environment: str,
    library_name: str | None,
    generate_before_dt: float,
    generate_after_dt: float,
    samples_before_urls: list[str],
    samples_after_urls: list[str],
):
    NEWLINE = '\n'
    IMAGE_EXTS = ('.png', '.webp', '.jpg', '.jpeg', '.gif')
    VIDEO_EXTS = ('.mp4', '.webm', '.mov')

    def media_cell(url: str):
        name = url.split('/')[-1]
        if name.endswith(IMAGE_EXTS):
            return f'![{name}]({url})'
        if name.endswith(VIDEO_EXTS):
            return f'<video src="{url}" controls></video>'
        return f'[{name}]({url})'

    return f"""
---
tags:
- ahead-of-time
- pytorch
- aokit
library_name: {library_name or 'pytorch'}
---

> [!NOTE]
> This **README** has been auto-generated by the **HF Job** run linked below
> and the whole repository is a reproducible artifact of this Job

# Ahead-of-time repository

AoT repos contain **pre-compiled binaries** of PyTorch models, packaged with
[`aokit`](https://github.com/huggingface/aokit), enabling:
- fast startup times (no `torch.compile` needed)
- significant **speedup**
- **ZeroGPU** compatibility

## How to use
``` python
{model_init}\n
{aokit_load}
```

## How to reproduce or customize
``` bash
# Install hf CLI
curl -LsSf https://hf.co/cli/install.sh | bash

# Login
hf auth login

# Get the job file and edit (user section) if needed
hf download {repo_id} job.py --local-dir .

# Run the job and change flavor or image if needed
hf jobs uv run job.py \\
    --flavor {job_flavor or DEFAULT_FLAVOR} \\
    --secrets HF_TOKEN

# Or run locally with Docker
docker run --rm --gpus all \\
    -v $PWD/job.py:/workspace/job.py \\
    -e HF_TOKEN=$(hf auth token) \\
    -e JOB_FLAVOR={job_flavor or DEFAULT_FLAVOR} \\
    ghcr.io/astral-sh/uv:python3.12-bookworm \\
    uv run /workspace/job.py
```

The following job [environment variables](https://hf.co/docs/hub/en/jobs-configuration#user-defined-environment-variables)
can be used to customize the repo name generation:
- `OUTPUT_REPO_NAMESPACE`: defaults to the authenticated user (`huggingface_hub.whoami()`)
- `OUTPUT_REPO_BASE_NAME`: defaults to the `module` class name
- `OUTPUT_REPO_ID`: fully overtakes name generation

## Samples
Generated as part of the compilation job: before and after compilation
| Before compilation ({generate_before_dt:.2f}s) | After compilation ({generate_after_dt:.2f}s) |
|------------------------------------------------|----------------------------------------------|
{NEWLINE.join(
    f"| {media_cell(before_url)} | {media_cell(after_url)} |"
    for before_url, after_url in zip(samples_before_urls, samples_after_urls)
)}

Speedup: **{generate_before_dt / generate_after_dt:.2f}x**
(note that this might not always reflect actual performance gain)

## Environment
<details>
<summary>Click to expand</summary>

```
{environment}
```
</details>

## Job run
{f'- [{job_id}](https://huggingface.co/jobs/{job_id})' if job_id is not None else '- (run locally)'}
"""
