# Using with HF Jobs

[HF Jobs](https://huggingface.co/docs/hub/en/jobs-overview) provides
an easy way to run different jobs on a variety of hardware. 

For AOT-compiled binaries to work properly, the execution environment must
match the environment in which the binaries were obtained. Users can
export the AOT-compiled binary on their choice of hardware through
HF Jobs and then deploy it later. This is particularly useful when
preparing [demos based on ZeroGPU](https://huggingface.co/docs/hub/en/spaces-zerogpu).
This is helpful because users can first obtain the binary in a
reproducible manner without killing precious ZeroGPU hours.

We provide an example of launching such as a job below. It exports
the AOT-compiled binaries for Flux.1-Dev and pushes them to the
Hub. It also generates samples with those binaries so that
users can verify them before the actual deployment.

<details>
<summary>Collapse</summary>

```py
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "aokit @ https://github.com/huggingface/aokit/archive/refs/heads/hf-jobs.tar.gz",  # aokit.hub_utils (branch tarball; no git CLI needed in the image)
#   "torch>=2.8.0",
#   "diffusers>=0.32.0",
#   "transformers",
#   "accelerate",
#   "sentencepiece",
#   "protobuf",
#   "huggingface_hub>=0.34.0",
#   "Pillow",
# ]
# ///
#
# Ahead-of-time compile a PyTorch module with `aokit`, run it on HF Jobs, and
# push the packaged binary (the `.pt2` artifact) to a repo on the Hugging Face Hub.
#
# This is a self-contained `uv` script (PEP 723 header above): all dependencies
# are declared inline, so it can be launched directly with the `hf` CLI.
#
# Run it on HF Jobs (defaults to the `rtx-pro-6000` flavor):
#
#     hf auth login
#     hf jobs uv run job.py \
#         --flavor rtx-pro-6000 \
#         --image pytorch/pytorch:2.9.1-cuda13.0-cudnn9-devel \
#         --secrets HF_TOKEN
#
# The `--image` is required: AOTInductor compiles a `.so`, which needs a full
# CUDA toolkit (nvcc + headers + `CUDA_HOME`). The default uv image ships only
# torch's CUDA *runtime* wheels, so use a CUDA *devel* image whose CUDA line
# matches torch's (here cuda13.0 <-> torch cu130). See `hub_utils.DEFAULT_IMAGE`.
#
# `--secrets HF_TOKEN` forwards your local token to the Job so it can push the
# resulting repo under your namespace. The output repo id is derived from
# `huggingface_hub.whoami()` and can be overridden with the `OUTPUT_REPO_*`
# environment variables documented in the generated README.
#
# The compile-and-publish orchestration lives in `aokit.hub_utils`; this script
# only supplies the model and the two handlers below. It defaults to the
# `rtx-pro-6000` flavor (see `aokit.hub_utils.DEFAULT_FLAVOR`).

from aokit.hub_utils import create_aoti_repo


# =========================
# User section
# =========================

# README::MODEL_INIT::START
import torch
from diffusers import FluxPipeline

pipeline = FluxPipeline.from_pretrained(
    'black-forest-labs/FLUX.1-dev',
    torch_dtype=torch.bfloat16,
).to('cuda')
# README::MODEL_INIT::END


def compile_and_save(module: torch.nn.Module, package_dir: str):
    """AoT compile `module` (a `FluxTransformer2DModel`) into `package_dir`.

    Flux is regionally compiled: one representative block of each stack is
    exported and compiled, then reused across every block in that stack at load
    time. The resulting weight-less `.pt2` archives live under
    `package_dir/submodules/<name>/`.
    """
    import aokit

    for submodule in ('transformer_blocks', 'single_transformer_blocks'):
        block = module.get_submodule(submodule)[0]

        # Capture example inputs by running the pipeline once (AoT needs shapes).
        with aokit.exporting.capture(block) as call:
            pipeline(prompt="prompt", num_inference_steps=1)

        with torch.no_grad():
            exported = torch.export.export(
                mod=block,
                args=call.args,
                kwargs=call.kwargs,
            )

        aokit.compile_and_save(
            package_dir=package_dir,
            exported_program=exported,
            submodule=submodule,
        )


def generate_samples(samples_dir: str):
    """Render a sample image from `pipeline` into `samples_dir`."""
    image = pipeline(
        prompt="realistic photo of a cat walking on the surface of the moon",
        guidance_scale=4.5,
        num_inference_steps=50,
        generator=torch.manual_seed(42),
        height=1024,
        width=1024,
    ).images[0]
    image.save(f'{samples_dir}/image.png')


def main():
    create_aoti_repo(
        module=pipeline.transformer,
        module_expr='pipeline.transformer',
        compile_and_save=compile_and_save,
        generate_samples=generate_samples,
    )


if __name__ == '__main__':
    main()

```

</details>

Run the script like so:

```bash
# provided the script is written in `job.py`
hf jobs uv run job.py \
    --flavor rtx-pro-6000 \
    --image pytorch/pytorch:2.9.1-cuda13.0-cudnn9-devel \
    --secrets HF_TOKEN
```

An example repo, containing the AOT binaries and the generated
samples, is available here: [sayakpaul/FluxTransformer2DModel-sm120-cu130-r54](https://huggingface.co/sayakpaul/FluxTransformer2DModel-sm120-cu130-r54).

