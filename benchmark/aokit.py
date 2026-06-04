# /// script
# requires-python = "==3.10.*"
# dependencies = [
#   "aokit==0.1.1",
#   "torch==2.12.0",
#   "torchvision",
#   "torchao==0.17.0",
#   "bitsandbytes",
#   "mslk @ https://download.pytorch.org/whl/nightly/cu130/mslk-2026.5.9%2Bcu130-cp310-cp310-manylinux_2_28_x86_64.whl",
#   "diffusers",
#   "ftfy",
#   "imageio-ffmpeg",
#   "imageio",
#   "huggingface_hub",
#   "transformers",
#   "accelerate",
#   "setuptools",
# ]
# ///

import time

import aokit
import torch
from PIL import Image
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.wan.pipeline_wan_i2v import WanImageToVideoPipeline
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from diffusers.quantizers import PipelineQuantizationConfig
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from torchao.quantization import quantize_
from torchao.quantization import Float8DynamicActivationFloat8WeightConfig


# Init
pipeline = WanImageToVideoPipeline.from_pretrained(
    'Wan-AI/Wan2.2-I2V-A14B-Diffusers',
    transformer=WanTransformer3DModel.from_pretrained('cbensimon/Wan2.2-I2V-A14B-bf16-Diffusers',
        subfolder='transformer',
        torch_dtype=torch.bfloat16,
        device_map='cuda',
    ),
    transformer_2=WanTransformer3DModel.from_pretrained('cbensimon/Wan2.2-I2V-A14B-bf16-Diffusers',
        subfolder='transformer_2',
        torch_dtype=torch.bfloat16,
        device_map='cuda',
    ),
    torch_dtype=torch.bfloat16,
    quantization_config=PipelineQuantizationConfig(
        quant_mapping={
            "text_encoder": TransformersBitsAndBytesConfig(load_in_8bit=True),
        }
    ),
)
pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipeline.scheduler.config, shift=8.0)
pipeline.to('cuda')

# Quantize
quantize_(pipeline.transformer.blocks, Float8DynamicActivationFloat8WeightConfig())
quantize_(pipeline.transformer_2.blocks, Float8DynamicActivationFloat8WeightConfig())

from diffusers.utils.loading_utils import load_image
image = load_image('https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/wan-cat.jpg')

# Compile and save
TRANSFORMER_SEQ_DIM = torch.export.Dim.DYNAMIC
TRANSFORMER_DYNAMIC_SHAPES = (
    {1: TRANSFORMER_SEQ_DIM},     # hidden_states: [B, S, D]
    None,                         # encoder_hidden_states
    None,                         # temb
    (
        {1: TRANSFORMER_SEQ_DIM}, # rotary_emb[0]: [1, S, 1, head_dim]
        {1: TRANSFORMER_SEQ_DIM}, # rotary_emb[1]: [1, S, 1, head_dim]
    ),
)

block = pipeline.transformer.blocks[0]
pacakge_dir = f'{block.__class__.__name__}/package'

with aokit.exporting.capture(block) as call:
    pipeline(
        image=Image.new("RGB", (640, 640)),
        prompt="prompt",
        width=640,
        height=640,
        num_frames=81,
    )

with torch.no_grad():
    exported = torch.export.export(
        mod=block,
        args=call.args,
        kwargs=call.kwargs,
        dynamic_shapes=TRANSFORMER_DYNAMIC_SHAPES,
    )

aokit.compile_and_save(
    package_dir=pacakge_dir,
    exported_program=exported,
    submodule='blocks',
)

# Load compiled
aokit.load_from_package_dir(pipeline.transformer, pacakge_dir)
aokit.load_from_package_dir(pipeline.transformer_2, pacakge_dir)

# Benchmark
image = load_image('https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/wan-cat.jpg')
prompt = "A chill cat taking a selfie while surfing on a windy day"
dts = []

# TODO: make the next section shared between all the benchmarks (eager, torch.compile, etc.)
# 1st video
t0 = time.perf_counter()
output = pipeline(
    image=image,
    prompt=prompt,
    width=624,
    height=832,
    num_frames=81,
    num_inference_steps=8,
    guidance_scale=1,
    guidance_scale_2=1,
    generator=torch.Generator(device='cuda').manual_seed(42),
)
dts += [-(t0 - (t0 := time.perf_counter()))]

# 2nd video
t0 = time.perf_counter()
output = pipeline(
    image=image,
    prompt=prompt,
    width=624,
    height=832,
    num_frames=81,
    num_inference_steps=8,
    guidance_scale=1,
    guidance_scale_2=1,
    generator=torch.Generator(device='cuda').manual_seed(42),
)
dts += [-(t0 - (t0 := time.perf_counter()))]

# 3rd video
t0 = time.perf_counter()
output = pipeline(
    image=image,
    prompt=prompt,
    width=640, # Changed width
    height=832,
    num_frames=81,
    num_inference_steps=8,
    guidance_scale=1,
    guidance_scale_2=1,
    generator=torch.Generator(device='cuda').manual_seed(42),
)
dts += [-(t0 - (t0 := time.perf_counter()))]
