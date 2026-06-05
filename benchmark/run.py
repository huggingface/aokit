# /// script
# requires-python = "==3.10.*"
# dependencies = [
#   "aokit==0.1.1",
#   "torch==2.12.0",
#   "torchvision",
#   "torchao==0.17.0",
#   "bitsandbytes",
#   "diffusers",
#   "ftfy",
#   "transformers",
#   "accelerate",
#   "setuptools",
# ]
# ///

import multiprocessing
import time

import aokit
import torch
from PIL import Image
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.wan.pipeline_wan_i2v import WanImageToVideoPipeline
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from diffusers.quantizers import PipelineQuantizationConfig
from diffusers.utils.loading_utils import load_image
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from torchao.quantization import quantize_
from torchao.quantization import Float8DynamicActivationFloat8WeightConfig


def load_pipeline():
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
    quantize_(pipeline.transformer.blocks, Float8DynamicActivationFloat8WeightConfig())
    quantize_(pipeline.transformer_2.blocks, Float8DynamicActivationFloat8WeightConfig())
    return pipeline


def aokit_compile(pipeline: WanImageToVideoPipeline, pacakge_dir: str):
    block = pipeline.transformer.blocks[0]
    with aokit.exporting.capture(block) as call:
        pipeline(
            image=Image.new("RGB", (640, 640)),
            prompt="prompt",
            width=640,
            height=640,
            num_frames=81,
        )
    dynadim_1 = {1: torch.export.Dim.DYNAMIC}
    with torch.no_grad():
        exported = torch.export.export(
            mod=block,
            args=call.args,
            kwargs=call.kwargs,
            dynamic_shapes=(dynadim_1, None, None, (dynadim_1, dynadim_1)),
        )
    aokit.compile_and_save(
        package_dir=pacakge_dir,
        exported_program=exported,
        submodule='blocks',
    )


def run_benchmark(pipeline: WanImageToVideoPipeline, name: str):
    image = load_image('https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/wan-cat.jpg')
    prompt = "A chill cat taking a selfie while surfing on a windy day"
    def run_generation(width: int):
        pipeline(
            image=image,
            prompt=prompt,
            width=width,
            height=832,
            num_frames=81,
            num_inference_steps=8,
            guidance_scale=1,
            guidance_scale_2=1,
            generator=torch.Generator(device='cuda').manual_seed(42),
        )
    t0 = time.perf_counter()
    run_generation(width=624)
    t1 = time.perf_counter()
    run_generation(width=624)
    t2 = time.perf_counter()
    run_generation(width=640)
    t3 = time.perf_counter()
    run_generation(width=656)
    t4 = time.perf_counter()
    with open(f'results-{name}.txt', 'wt') as f:
        f.write(f'{t1-t0}\n')
        f.write(f'{t2-t1}\n')
        f.write(f'{t3-t2}\n')
        f.write(f'{t4-t3}\n')


def run_benchmark_eager():
    pipeline = load_pipeline()
    run_benchmark(pipeline, 'eager')


def run_benchmark_compile():
    pipeline = load_pipeline()
    pipeline.transformer.compile_repeated_blocks()
    pipeline.transformer_2.compile_repeated_blocks()
    run_benchmark(pipeline, 'compile')


def run_benchmark_aokit():
    pipeline = load_pipeline()
    aokit_compile(pipeline, 'Wan2Transformer-package')
    aokit.load_from_package_dir(pipeline.transformer, 'Wan2Transformer-package')
    aokit.load_from_package_dir(pipeline.transformer_2, 'Wan2Transformer-package')
    run_benchmark(pipeline, 'aokit')


def main():
    mp_context = multiprocessing.get_context('forkserver')
    for target in (
        run_benchmark_eager,
        run_benchmark_compile,
        run_benchmark_aokit
    ):
        p = mp_context.Process(target=target)
        p.start()
        p.join()
        assert p.exitcode == 0


if __name__ == '__main__':
    main()
