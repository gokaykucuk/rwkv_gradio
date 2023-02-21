from rwkvstic.agnostic.backends import TORCH, TORCH_QUANT
import torch

quantized = {
    "mode": TORCH_QUANT,
    "runtimedtype": torch.bfloat16,
    "chunksize": 32,  # larger = more accurate, but more memory (and slower)
    "target": 24  # your gpu max size, excess vram offloaded to cpu
}


config = {
    "path": "models/concular_v2_only_notion.pth",
    "mode": TORCH,
    "runtimedtype": torch.float32,
    "useGPU": torch.cuda.is_available(),
    "dtype": torch.float32
}

config = {
    "path": "models/concular_v2_only_notion.pth",
    **quantized
}

title = "Bot UI"
