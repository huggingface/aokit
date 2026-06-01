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
from diffusers import Flux2KleinPipeline
from diffusers.utils import load_image
import torch

pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B",
    torch_dtype=torch.bfloat16
).to("cuda")

url = "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/wan-cat.jpg"
image = pipe(
    prompt="Remove the sunglasses",
    image=load_image(url).resize((1024, 1024)),
    guidance_scale=2.5,
    generator=torch.manual_seed(42),
).images[0]
image.save("generated_image.png")
```

The changes are best described with a diff:

```diff
from diffusers import Flux2KleinPipeline
+ from aokit.exporting import capture
+ from PIL import Image
import torch
+ import aokit

pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B",
    torch_dtype=torch.bfloat16
).to("cuda")

+ # Capture example inputs of the `transformer` because this is AOT
+ with capture(pipe.transformer) as call:
+    pipe(
+        prompt="prompt", image=[Image.new("RGB", (1024, 1024))],
+    )

+ # Export the module to a program
+ with torch.no_grad():
+    exported = torch.export.export(
+        mod=pipe.transformer,
+        args=call.args,
+        kwargs=call.kwargs,
+    )

+ # Perform compilation and serialize as the graph as a binary
+ package_dir = "flux2_klein_exported"
+ aokit.compile_and_save(
+    package_dir=package_dir, exported_program=exported,
+)
```

Now, use the compiled binary and run inference:

```diff
from diffusers.utils import load_image

+ aokit.load_from_module_dir(pipe.transformer, f"{package_dir}/root")

url = "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/wan-cat.jpg"
image = pipe(
    prompt="Remove the sunglasses",
    image=load_image(url).resize((1024, 1024)),
    guidance_scale=2.5,
    generator=torch.manual_seed(42),
).images[0]
image.save("generated_image.png")
```

Here is the full snippet end-to-end:

<details>
<summary>Collapse</summary>

```py
from diffusers import Flux2KleinPipeline
from diffusers.utils import load_image
from aokit.exporting import capture
from PIL import Image
import torch
import aokit

# Load an image generation model
pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B",
    torch_dtype=torch.bfloat16
).to("cuda")

# Capture example inputs of the `transformer` because this is AOT
with capture(pipe.transformer) as call:
    pipe(
        prompt="prompt",
        image=[Image.new("RGB", (1024, 1024))],
    )

# Export the module to a program
with torch.no_grad():
    exported = torch.export.export(
        mod=pipe.transformer,
        args=call.args,
        kwargs=call.kwargs,
    )

# Perform compilation and serialize as the graph as a binary
package_dir = "flux2_klein_exported"
aokit.compile_and_save(
    package_dir=package_dir,
    exported_program=exported,
)

aokit.load_from_module_dir(pipe.transformer, f"{package_dir}/root")

url = "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/wan-cat.jpg"
image = pipe(
    prompt="Remove the sunglasses",
    image=load_image(url).resize((1024, 1024)),
    guidance_scale=2.5,
    generator=torch.manual_seed(42),
).images[0]
image.save("generated_image.png")
```

</details>

## Dynamic shapes

## Regional compilation

## Compilation flags

## Kernels support

## API documentation

Coming soon
