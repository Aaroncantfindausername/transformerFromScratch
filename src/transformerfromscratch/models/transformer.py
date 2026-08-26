import torch
from torch import float32, nn


class SelfAttention(nn.Module):
    tril: torch.Tensor

    def __init__(
        self,
        d_model: int,
        d_q: int,
        seq_len: int,
        batch_size: int,
        dropout: float,
        device: str | torch.device,
    ):
        super().__init__()
        self.query_weights = nn.Linear(d_model, d_q, bias=False, device=device)
        self.key_weights = nn.Linear(d_model, d_q, bias=False, device=device)
        self.value_weights = nn.Linear(
            d_model, d_model, bias=False, device=device
        )
        self.d_q = d_q
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "tril", torch.tril(torch.ones((seq_len, seq_len), device=device))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, S, d) (Batch, seq_len, d_model)
        q = self.query_weights(x)  # (B, S, d_qk)
        k = self.key_weights(x)  # (B, S, d_qk)
        v = self.value_weights(x)  # (B, S, d_model)
        attention_scores = q @ torch.transpose(k, 1, 2)  # (B, S, S)
        attention_scores = torch.masked_fill(
            attention_scores, ~self.tril, float("-inf")
        )  # (B, S, S)
        attention_scores = nn.functional.softmax(
            attention_scores * self.d_q ** (-1 / 2), dim=0
        )
        attention_scores = self.dropout(attention_scores)
        return attention_scores @ v  # (B, S, d_model)


class MultiHeadedSelfAttention(nn.Module):
    tril: torch.Tensor

    def __init__(
        self,
        d_model: int,
        d_q: int,
        seq_len: int,
        batch_size: int,
        dropout: float,
        n_heads: int,
        device: str | torch.device,
        flash_attention: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0 and d_q % n_heads == 0, (
            "d_model and d_q should be divisible by n.o attention heads"
        )
        self.n_heads = n_heads
        self.d_q = d_q
        self.d_model = d_model
        self.qkv_weights = nn.Linear(d_model, 2 * d_q + d_model)
        self.flash = flash_attention
        self.register_buffer(
            "tril", torch.tril(torch.ones((seq_len, seq_len), device=device))
        )
        self.p_dropout = dropout
        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, S, d_model) (Batch, seq_len, d_model)
        B, S, _ = x.shape
        q, k, v = torch.split(
            self.qkv_weights(x), [self.d_q, self.d_q, self.d_model]
        )
        # q,k : (B, S, d_q), v: (B, S, d_model)
        q = q.view(
            B, S, self.n_heads, self.d_q // self.n_heads
        )  # B,S, n_heads, d_q_head
        k = k.view(
            B, S, self.n_heads, self.d_q // self.n_heads
        )  # B,S, n_heads, d_q_head
        v = v.view(
            B, S, self.n_heads, self.d_model // self.n_heads
        )  # B,S, n_heads, d_model_head
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                dropout_p=self.p_dropout if self.tranining else 0,
                is_causal=True,
            )
            y = y.transpose(1, 2)
        else:
            attention_scores = torch.einsum(
                "BShd, BThd -> BhST", q, k
            ) / q.shape[-1] ** (0.5)  # B,n_heads, S, S
            attention_scores = attention_scores.masked_fill(
                ~self.tril, float("-inf")
            )
            attention_scores = torch.softmax(attention_scores, dim=-1)
            attention_scores = self.attention_dropout(attention_scores)
            y = torch.einsum(
                "BhSS, BShd -> BShd",
                attention_scores,
            )
        y = y.reshape(B, S, self.d_model)
        y = self.residual_dropout(y)
        return y


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        qk_dim: int,
        seq_len: int,
        batch_size: int,
        attention_heads: int,
        ff_hidden_dim: int,
        ff_layers: int,
        device: str | torch.device,
    ) -> None:
        super().__init__()
