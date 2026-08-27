from transformerfromscratch.models.transformer import MultiHeadedSelfAttention
import torch
import numpy as np

torch.manual_seed(67)
np.random.seed(67)
d_model = 8
d_q = 4
n_heads = 2
batch_size = 1
seq_len = 3
dropout = 0.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiHeadedSelfAttention(
    d_model, d_q, seq_len, batch_size, dropout, n_heads, device, False
)
input = torch.tensor(
    [1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0],
    dtype=torch.float32,
    device=device,
).reshape(batch_size, seq_len, d_model)
model.eval()
output = model(input)
pass
