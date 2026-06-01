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

Coming soon

## API documentation

Refer to [`src/aokit/aokit.py`](./src/aokit/aokit.py).
