import torch
from torch.utils.data import TensorDataset

from transformerfromscratch.pipeline.data import (
    encode_text,
    load_data,
    train_val_test_split,
)


text = load_data("datasets/tiny_shakespear.txt")
encoded_text, decoder = encode_text(text)
context_length = 8
batch_size = 64
train_loader, val_loader, test_loader = train_val_test_split(
    encoded_text, context_length=context_length, batch_size=batch_size
)
