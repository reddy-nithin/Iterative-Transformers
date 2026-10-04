"""Parameter count, FLOPs, throughput."""
import time

import torch
from torch.utils.flop_counter import FlopCounterMode


def count_params(model) -> int:
    """Unique trainable parameters (shared weights counted once)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def count_gflops(model, device, img_size=224) -> float:
    """Forward GFLOPs for one image (matmul/conv ops; 1 MAC = 2 FLOPs)."""
    model.eval()
    x = torch.randn(1, 3, img_size, img_size, device=device)
    with FlopCounterMode(display=False) as fc:
        model(x)
    return fc.get_total_flops() / 1e9


@torch.no_grad()
def measure_throughput(model, device, batch_size=256, img_size=224, warmup=10, iters=30) -> float:
    """Images/second for inference (fp16/bf16 autocast on CUDA)."""
    model.eval()
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    use_amp = device.type == "cuda"
    sync = torch.cuda.synchronize if use_amp else (lambda: None)
    with torch.autocast(device.type, enabled=use_amp):
        for _ in range(warmup):
            model(x)
        sync()
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        sync()
    return batch_size * iters / (time.perf_counter() - t0)
