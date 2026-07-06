"""
Instruction-tuning dataset for the chatbot.

Each (question, answer) pair becomes one training sequence:

    <user>  q0 q1 ... qn  <bot>  a0 a1 ... am  <eos>

The model is trained to predict the next token, but -- crucially -- the loss is
**masked to the answer only**. We do not want the model to spend capacity
learning to reproduce the user's questions or the padding; we only reward it for
producing the right answer (and the <eos> that ends it). Masked target positions
are set to -1, which src/model/gpt.py's cross-entropy ignores.

Sequences are padded to `block_size` so they can be stacked into batches.
"""

import torch
from torch.utils.data import Dataset


class ChatDataset(Dataset):
    def __init__(self, pairs, tokenizer, block_size: int):
        self.examples = []
        pad_id = tokenizer.stoi["<pad>"]
        user_id = tokenizer.stoi["<user>"]
        bot_id = tokenizer.stoi["<bot>"]
        eos_id = tokenizer.stoi["<eos>"]

        for q, a in pairs:
            q_ids = tokenizer.encode(q)
            a_ids = tokenizer.encode(a)
            seq = [user_id] + q_ids + [bot_id] + a_ids + [eos_id]
            seq = seq[: block_size + 1]  # need block_size inputs + 1 shifted target

            x = seq[:-1]
            y = seq[1:]

            # Index of <bot> within `seq`; everything the model should learn to
            # produce (the answer + <eos>) starts right after it. In the shifted
            # target `y`, those live at indices >= bot_index.
            bot_index = 1 + len(q_ids)
            y = [(tok if i >= bot_index else -1) for i, tok in enumerate(y)]

            # right-pad to block_size (padded inputs -> <pad>, padded targets -> -1)
            pad = block_size - len(x)
            x = x + [pad_id] * pad
            y = y + [-1] * pad

            self.examples.append(
                (torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long))
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
