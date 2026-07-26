"""
Dataset for next-token prediction.

Given a long stream of token ids, every training example is a `block_size`-length
window `x` paired with the same window shifted one position to the right, `y`.
The model learns to predict, for every position in `x`, the token that comes next.

Example with block_size=4 on the id sequence [10, 11, 12, 13, 14, 15]:
    x = [10, 11, 12, 13]
    y = [11, 12, 13, 14]
This single example actually teaches the model 4 predictions at once: given "10",
predict "11"; given "10, 11", predict "12"; and so on.
"""

import torch
from torch.utils.data import Dataset


class CharDataset(Dataset):
    def __init__(self, data: torch.Tensor, block_size: int):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx):
        # Convert to int64 here: the data may be stored as int16 to save memory,
        # but embedding lookups and cross-entropy need long indices.
        chunk = self.data[idx: idx + self.block_size + 1].long()
        x = chunk[:-1]
        y = chunk[1:]
        return x, y
