# AOKit

### Ahead-of-time compilation toolkit

`aokit` is a lightweight set of tools sitting on top of PyTorch ahead-of-time compilation APIs (e.g. `torch.export.export` and `torch._inductor.aot*`) to help organize, package, and distribute AoT artifacts within the Hugging Face ecosystem.

Extending PyTorch [AOTInductor](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_aot_inductor.html), `aokit` provides:

- **Regional compilation**: compiles and loads individual submodules independently, with support for dynamic weight swapping even after quantization (e.g. with [`torchao`](https://github.com/pytorch/ao))
- **Weight-less packaging**: produces lightweight, composable, and reusable artifacts while decoupling graph and weights management
- **[ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu) compatibility**: leverages lazy execution for seamless integration
- **[`kernels`](https://github.com/huggingface/kernels) integration**: automatically packages and loads required kernels (coming soon)

## Installation

``` bash
pip install aokit
```

## Quickstart

We recommend following the [accompanying blog post](https://huggingface.co/blog/zerogpu-aoti).

Using AOT-compiled models in the context of Diffusers is just a few lines of code. Normally,
one would run inference in Diffusers like so:

```py
from diffusers import FluxPipeline
import torch

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
).to("cuda")

image = pipe(
    prompt="realistic photo a cat walking on the surface of moon",
    guidance_scale=4.5,
    num_inference_steps=50,
    generator=torch.manual_seed(42),
).images[0]
image.save("generated_image.png")
```

The changes are best described with a diff:

```diff
from diffusers import FluxPipeline
+ import aokit

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
).to("cuda")

+ with aokit.exporting.capture(pipe.transformer) as call:
+    pipe(prompt="prompt")

+ with torch.no_grad():
+    exported = torch.export.export(
+        mod=pipe.transformer,
+        args=call.args,
+        kwargs=call.kwargs,
+    )

+ package_dir = "flux_exported"
+ aokit.compile_and_save(
+    package_dir=package_dir, exported_program=exported,
+)
```

Now, use the compiled binary and run inference:

```diff
+ aokit.load_from_package_dir(pipe.transformer, package_dir)

image = pipe(
    prompt="realistic photo a cat walking on the surface of moon",
    guidance_scale=4.5,
    num_inference_steps=50,
    generator=torch.manual_seed(42),
).images[0]
image.save("generated_image.png")
```

Here is the full snippet end-to-end (with comments):

<details>
<summary>Collapse</summary>

```py
from diffusers import FluxPipeline
import torch
import aokit

# Load an image generation model
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
).to("cuda")

# Capture example inputs of the `transformer` because this is AOT
with aokit.exporting.capture(pipe.transformer) as call:
    pipe(prompt="prompt")

# Export the module to a program
with torch.no_grad():
    exported = torch.export.export(
        mod=pipe.transformer,
        args=call.args,
        kwargs=call.kwargs,
    )

# Perform compilation and serialize as the graph as a binary
package_dir = "flux_exported"
aokit.compile_and_save(
    package_dir=package_dir,
    exported_program=exported,
)

aokit.load_from_package_dir(pipe.transformer, package_dir)

image = pipe(
    prompt="realistic photo a cat walking on the surface of moon",
    guidance_scale=4.5,
    num_inference_steps=50,
    generator=torch.manual_seed(42),
).images[0]
image.save("generated_image.png")
```

</details>

## Guides

You can find several other guides at [`guides`](./guides/):

* [Dynamic shapes](./guides/dynamic_shapes.md)
* [Regional compilation](./guides/regional_compilation.md)
* Compilation flags (coming)
* Kernels support (coming)
* [Using with HF Jobs](./guides/hf-jobs.md)

## API documentation

Refer to [`src/aokit/aokit.py`](./src/aokit/aokit.py).
