# Quantization

It is possible to apply quantization before exporting the binary and then
load it later for inference. `aokit` is agnostic to the quantization
backend being used. So, users have the flexibility to choose a backend
that best performs on their setup. This guide will use [TorchAO](https://github.com/pytorch/ao).

The workflow is to first apply quantization _before_ exporting the binary.
Alternatively, users can also load quantized checkpoints. In Diffusers,
we support TorchAO natively as a quantization backend. In terms of
code changes, this would be:

```diff
- from diffusers import FluxPipeline
+ from diffusers import FluxPipeline, TorchAoConfig, PipelineQuantizationConfig
+ from torchao.quantization import Float8DynamicActivationFloat8WeightConfig

+ pipeline_quant_config = PipelineQuantizationConfig(
+    quant_mapping={"transformer": TorchAoConfig(Float8DynamicActivationFloat8WeightConfig())}
+)
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
+    quantization_config=pipeline_quant_config,
    torch_dtype=torch.bfloat16,
).to("cuda")

...
```

In case, there is no native integration available, users can still
leverage the `quantize_()` method.

```diff
+ from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig
...

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
).to("cuda")

+ quantize_(
+    pipe.transformer, Float8DynamicActivationFloat8WeightConfig()
+)
...
```

Find an end-to-end working example below.

<details>
<summary>Collapse</summary>

```py
from diffusers import FluxPipeline, TorchAoConfig, PipelineQuantizationConfig
from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
import torch
import aokit

# Load an image generation model, quantizing the transformer to Float8 with TorchAO.
# `Float8DynamicActivationFloat8WeightConfig` stores the weights in float8 and
# quantizes activations on the fly (best FP8 speed/memory/quality trade-off on
# GPUs with compute capability >= 8.9, e.g. RTX-4090 / Hopper).
pipeline_quant_config = PipelineQuantizationConfig(
    quant_mapping={"transformer": TorchAoConfig(Float8DynamicActivationFloat8WeightConfig())}
)
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    quantization_config=pipeline_quant_config,
    torch_dtype=torch.bfloat16,
).to("cuda")

# Capture example inputs of the (now quantized) `transformer` because this is AOT
with aokit.exporting.capture(pipe.transformer) as call:
    pipe(prompt="prompt")

# Export the quantized module to a program
with torch.no_grad():
    exported = torch.export.export(
        mod=pipe.transformer,
        args=call.args,
        kwargs=call.kwargs,
    )

# Perform compilation and serialize the graph as a binary
package_dir = "flux_float8_exported"
aokit.compile_and_save(
    package_dir=package_dir,
    exported_program=exported,
)

# Load the exported quantized binary back into the transformer. `aokit.patch`
# unwraps the torchao tensor-subclass parameters so the float8 weights are bound
# to the compiled model.
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

> [!TIP]
> Quantization also works with other features like shape dynamism, 
> regional compilation, and kernels integration.