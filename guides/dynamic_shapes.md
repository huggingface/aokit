# AOT with Dynamic Shapes

The quickstart example showed how `aokit` can be quickly
used to compile a model ahead-of-time and to run inference
with the AOT-compiled binary.

The example assumed that the image resolution will remain
static. However, realistically, we need to be able to generate
images of varied resolutions. To allow the compiler cater to
dynamic shapes, we need to:

* Define the axes of the input parameters / arguments that can
be dynamic.
* Create a dictionary mapping the parameter / argument names to
their axis-level shape specifications.
* Pass the dictionary to `torch.export.export()`.

Let's see the main changes we need to make to the example above:

```diff
from diffusers import FluxPipeline
import aokit

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
).to("cuda")

+ transformer_hidden_dim = torch.export.Dim("hidden", min=4096, max=8212)
+ transformer_dynamic_shapes = {
+    "hidden_states": {1: transformer_hidden_dim}, 
+    "img_ids": {0: transformer_hidden_dim},
}

with aokit.exporting.capture(pipe.transformer) as call:
    pipe(prompt="prompt")

+ dynamic_shapes = tree_map(lambda v: None, call.kwargs)
+ dynamic_shapes |= transformer_dynamic_shapes

with torch.no_grad():
    exported = torch.export.export(
        mod=pipe.transformer,
        args=call.args,
        kwargs=call.kwargs,
+       dynamic_shapes=dynamic_shapes
    )
```

Refer to the full example below:

<details>
<summary>Collapse</summary>

```py
from diffusers import FluxPipeline
import torch
import aokit
from torch.utils._pytree import tree_map

# Load an image generation model
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
).to("cuda")

# Define the argument names that can get impacted because
# of varied image resolution.
transformer_hidden_dim = torch.export.Dim('hidden', min=4096, max=8212)
transformer_dynamic_shapes = {
    "hidden_states": {1: transformer_hidden_dim}, 
    "img_ids": {0: transformer_hidden_dim},
}

# Capture example inputs of the `transformer` because this is AOT
with aokit.exporting.capture(pipe.transformer) as call:
    pipe(prompt="prompt")

# Only change the argument defined above but the dictionary
# has to contain the accepted args in `call`.
dynamic_shapes = tree_map(lambda v: None, call.kwargs)
dynamic_shapes |= transformer_dynamic_shapes

# Export the module to a program with dynamic shapes specified.
with torch.no_grad():
    exported = torch.export.export(
        mod=pipe.transformer,
        args=call.args,
        kwargs=call.kwargs,
        dynamic_shapes=dynamic_shapes
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
    height=1024,
    width=1152
).images[0]
image.save("generated_image.png")
```

</details>

**Notes**:

* Configuring dynamism in AOT compilation needs a bit of
manual inspection. It can vary from model to model.
* The example above only works for varied input resolutions
(supported by the Flux.1-Dev model). It will still fail if a
different batch size is provided. We courage users to figure
out how that can be mitigated. 

<details>
<summary>Below is one solution</summary>

```py
from diffusers import FluxPipeline
import torch
import aokit
from torch.utils._pytree import tree_map

# Load an image generation model
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
).to("cuda")

# Define the argument names that can get impacted because
# of varied image resolution.
transformer_hidden_dim = torch.export.Dim('hidden', min=4096, max=8212)
transformer_dynamic_shapes = {
    "hidden_states": {0: torch.export.Dim.AUTO, 1: transformer_hidden_dim}, 
    "encoder_hidden_states": {0: torch.export.Dim.AUTO},
    "img_ids": {0: transformer_hidden_dim},
}

# Capture example inputs of the `transformer` because this is AOT
with aokit.exporting.capture(pipe.transformer) as call:
    pipe(prompt="prompt")

# Only change the argument defined above but the dictionary
# has to contain the accepted args in `call`.
dynamic_shapes = tree_map(lambda v: None, call.kwargs)
dynamic_shapes |= transformer_dynamic_shapes

# Export the module to a program with dynamic shapes specified.
with torch.no_grad():
    exported = torch.export.export(
        mod=pipe.transformer,
        args=call.args,
        kwargs=call.kwargs,
        dynamic_shapes=dynamic_shapes
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
    height=1024,
    width=1152,
    num_images_per_prompt=4
).images[0]
image.save("generated_image.png")
```

</details>