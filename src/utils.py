"""Small shared helpers used by the training and generation scripts."""

import math
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed every source of randomness so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def quantize_tensor_int8(t):
    """Per-row symmetric int8 quantization of a weight tensor.

    Each row (output feature) gets its own scale = max(|row|)/127, so one row
    with large values doesn't wreck the precision of the others. Returns the
    flat int8 values, the per-row float32 scales, the number of rows, and the
    dequantized tensor (q * scale) for verifying quality. Halves the on-disk
    size vs float16 and shrinks it ~4x vs float32 JSON.
    """
    w = t.detach().cpu().float()
    shape = list(w.shape)
    w2 = w.reshape(1, -1) if w.dim() <= 1 else w.reshape(w.shape[0], -1)
    absmax = w2.abs().amax(dim=1)
    scale = (absmax / 127.0).clamp(min=1e-8)
    q = torch.round(w2 / scale[:, None]).clamp(-127, 127).to(torch.int8)
    deq = (q.float() * scale[:, None]).reshape(shape)
    nrows = 1 if len(shape) <= 1 else shape[0]
    return q.reshape(-1).numpy(), scale.numpy().astype("<f4"), nrows, deq


def quantize_model_int8_(model):
    """Round every parameter to its int8-quantized value in place. Used to
    measure how much accuracy int8 costs before shipping it."""
    with torch.no_grad():
        for p in model.parameters():
            _, _, _, deq = quantize_tensor_int8(p.data)
            p.data = deq.to(p.data.dtype)


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """Linear warmup followed by cosine decay -- the standard GPT-style learning-rate schedule.

    1. For the first `warmup_steps`, the learning rate ramps up linearly from 0 to `max_lr`.
       This avoids destabilizing the randomly-initialized weights with a large update early on.
    2. After warmup, it decays following a cosine curve down to `min_lr` by `max_steps`.
    """
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, (max_steps - warmup_steps))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # ranges 1 -> 0
    return min_lr + coeff * (max_lr - min_lr)
