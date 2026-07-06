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
