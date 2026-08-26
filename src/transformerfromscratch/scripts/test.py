from transformerfromscratch.models.transformer import MultiHeadedSelfAttention
import torch

d_model = 8
d_q = 4
n_heads = 2
batch_size = 1
seq_len = 3
dropout = 0.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MultiHeadedSelfAttention(
    d_model, d_q, seq_len, batch_size, dropout, n_heads, device, False
)
