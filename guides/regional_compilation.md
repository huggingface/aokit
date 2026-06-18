# AOT with regional compilation

> [!NOTE]
> This guide assumes that the reader has gone through the
> [quickstart example](../README.md#quickstart) in the REAMDE.

Depending on the size of a model, full compilation can be
an expensive process. For models, composed of repeated
blocks, it can often be beneficial to _just_ compile
those repeated blocks, instead. This idea is referred to
as "regional compilation" (since we are compiling specific
"regions" within a model graph).

The rationale here is that those blocks are usually the
most compute-intensive blocks within the model. So,
compiling a single block and reusing it for the rest
(since these blocks share the same compute graph), can
speed up the compilation process while still delivering
latency improvements.

To do this with `aokit`, we first need to identify the
repeated blocks within model being AOT-compiled.

For the Flux Diffusion Transformer ([`FluxTransformer2DModel`](https://github.com/huggingface/diffusers/blob/9a72cd3ee9eaefbf5cac47640ba1c3acf082634d/src/diffusers/models/transformers/transformer_flux.py#L523)), there
are [two repeated blocks](https://github.com/huggingface/diffusers/blob/9a72cd3ee9eaefbf5cac47640ba1c3acf082634d/src/diffusers/models/transformers/transformer_flux.py#L608-L628):

* `transformer_blocks`
* `single_transformer_blocks`

Instead of compiling the full `pipe.transformer`, we now
have to compile a _single_ module within the block, like so:

```py
with aokit.exporting.capture(pipe.transformer.transformer_blocks[0]) as call_block_one:
    pipe(prompt="prompt")

with aokit.exporting.capture(pipe.transformer.single_transformer_blocks[0]) as call_block_two:
    pipe(prompt="prompt")

with torch.no_grad():
    exported_block_one = torch.export.export(
        mod=pipe.transformer.transformer_blocks[0],
        args=call_block_one.args,
        kwargs=call_block_one.kwargs,
    )
    exported_block_two = torch.export.export(
        mod=pipe.transformer.single_transformer_blocks[0],
        args=call_block_two.args,
        kwargs=call_block_two.kwargs,
    )

package_dir = "flux_exported"
aokit.compile_and_save(package_dir=package_dir, exported_program=exported_block_one, submodule="transformer_blocks")
aokit.compile_and_save(
    package_dir=package_dir, 
    exported_program=exported_block_two, 
    submodule="single_transformer_blocks"
)
```

Once these submodules are saved, we should expect to see:

```bash
$ ls flux_exported/
submodules
$ ls flux_exported/submodules/
transformer_blocks  single_transformer_blocks
```

We can use the same `load_from_package_dir()` and point
it towards `package_dir`. It will detect and load any
submodules automatically if they are present.

```py
aokit.load_from_package_dir(pipe.transformer, package_dir)

image = pipe(
    prompt="realistic photo a cat walking on the surface of moon",
    guidance_scale=4.5,
    num_inference_steps=50,
    generator=torch.manual_seed(42),
    height=1024,
    width=1024,
    num_images_per_prompt=4,
).images[0]
image.save("generated_image.png")
```

Find the fully working example below:

<details>
<summary>Collapse</summary>

```py
from diffusers import FluxPipeline
import torch
import aokit

# Load an image generation model
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda")

# Capture example inputs of the `transformer` because this is AOT
with aokit.exporting.capture(pipe.transformer.transformer_blocks[0]) as call_block_one:
    pipe(prompt="prompt")

with aokit.exporting.capture(pipe.transformer.single_transformer_blocks[0]) as call_block_two:
    pipe(prompt="prompt")

with torch.no_grad():
    exported_block_one = torch.export.export(
        mod=pipe.transformer.transformer_blocks[0],
        args=call_block_one.args,
        kwargs=call_block_one.kwargs,
    )
    exported_block_two = torch.export.export(
        mod=pipe.transformer.single_transformer_blocks[0],
        args=call_block_two.args,
        kwargs=call_block_two.kwargs,
    )

# Perform compilation and serialize as the graph as a binary
package_dir = "flux_exported"
aokit.compile_and_save(package_dir=package_dir, exported_program=exported_block_one, submodule="transformer_blocks")
aokit.compile_and_save(
    package_dir=package_dir, exported_program=exported_block_two, submodule="single_transformer_blocks"
)

aokit.load_from_package_dir(pipe.transformer, package_dir)

image = pipe(
    prompt="realistic photo a cat walking on the surface of moon",
    guidance_scale=4.5,
    num_inference_steps=50,
    generator=torch.manual_seed(42),
    height=1024,
    width=1024,
    num_images_per_prompt=4,
).images[0]
image.save("generated_image.png")

```

</details>

To get a sense how of the speed-compilation trade-off
between full compilation and regional compilation,
check out [this resource](https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/).

## Resources

* [Reducing torch.compile cold start compilation time with regional compilation](https://docs.pytorch.org/tutorials/recipes/regional_compilation.html)
* [Reducing AoT cold start compilation time with regional compilation](https://docs.pytorch.org/tutorials/recipes/regional_aot.html)