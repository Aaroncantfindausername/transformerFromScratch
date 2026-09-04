import torch
from torch import nn
from torch.nn.modules import LayerNorm


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
        self.out_proj = nn.Linear(d_model, d_model, bias=False, device=device)

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
        return self.out_proj(attention_scores @ v)  # (B, S, d_model)


class MultiHeadedSelfAttention(nn.Module):
    tril: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor

    def __init__(
        self,
        d_model: int,
        d_q: int,
        seq_len: int,
        batch_size: int,
        dropout: float,
        n_heads: int,
        device: str | torch.device,
        rope_freq_base: int = 10000,
        flash_attention: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0 and d_q % n_heads == 0, (
            "d_model and d_q should be divisible by n.o attention heads"
        )
        self.n_heads = n_heads
        self.d_q = d_q
        self.d_model = d_model
        self.qkv_weights = nn.Linear(d_model, 2 * d_q + d_model, device=device)
        self.flash = flash_attention
        self.register_buffer(
            "tril", torch.tril(torch.ones((seq_len, seq_len), device=device))
        )
        d = torch.arange(d_q // n_heads // 2).unsqueeze(0)  # (1, d_head/2)
        s = torch.arange(seq_len).unsqueeze(1)  # (S, 1)
        self.register_buffer(
            "cos", torch.cos(s * rope_freq_base ** (-d / d_model)).to(device)
        )
        self.register_buffer(
            "sin", torch.sin(s * rope_freq_base ** (-d / d_model)).to(device)
        )
        self.p_dropout = dropout
        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, S, d_model) (Batch, seq_len, d_model)
        B, S, _ = x.shape
        q, k, v = torch.split(
            self.qkv_weights(x), [self.d_q, self.d_q, self.d_model], dim=-1
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
        q = apply_rope(q, self.cos[:S, :], self.sin[:S, :])
        k = apply_rope(k, self.cos[:S, :], self.sin[:S, :])
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                dropout_p=self.p_dropout if self.training else 0,
                is_causal=True,
            )
            y = y.transpose(1, 2)
        else:
            attention_scores = torch.einsum(
                "BShd, BThd -> BhST", q, k
            ) / q.shape[-1] ** (0.5)  # B,n_heads, S, S
            attention_scores = attention_scores.masked_fill(
                self.tril == 0, float("-inf")
            )
            attention_scores = torch.softmax(attention_scores, dim=-1)
            attention_scores = self.attention_dropout(attention_scores)
            y = torch.einsum("BhST, BThd -> BShd", attention_scores, v)
        y = y.reshape(B, S, self.d_model)
        y = self.residual_dropout(y)
        return y


class MLP(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        activation: str,
        p_dropout: float,
        device: str | torch.device,
    ):
        super().__init__()
        self.activation = activation
        layers = []
        prev_out_dim = d_model
        for i in range(n_layers):
            next_dim = (
                prev_out_dim * 2 if i < n_layers // 2 else prev_out_dim // 2
            )
            layers.append(nn.Linear(prev_out_dim, next_dim, device=device))
            layers.append(self.get_activation())
            layers.append(nn.Dropout(p_dropout))
            prev_out_dim = next_dim
        layers.append(nn.Linear(prev_out_dim, d_model, device=device))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x :(B, S, d)
        return self.net(x)

    def get_activation(self) -> nn.Module:
        if self.activation == "relu":
            return nn.ReLU()
        elif self.activation == "gelu":
            return nn.GELU()
        else:
            raise ValueError


class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_q: int,
        dropout: float,
        seq_len: int,
        batch_size: int,
        mlp_layers: int,
        mlp_activation: str,
        n_attn_heads: int,
        device: str | torch.device,
    ) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(d_model, device=device)
        self.ln_2 = LayerNorm(d_model, device=device)
        self.attention = MultiHeadedSelfAttention(
            d_model,
            d_q,
            seq_len,
            batch_size,
            dropout,
            n_attn_heads,
            device,
            flash_attention=True,
        )
        self.mlp = MLP(d_model, mlp_layers, mlp_activation, dropout, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_q: int,
        dropout: float,
        context_window: int,
        batch_size: int,
        mlp_layers: int,
        mlp_activation: str,
        n_attn_heads: int,
        n_blocks: int,
        vocab_size: int,
        device: str | torch.device,
    ):
        super().__init__()
        self.context_window = context_window
        self.token_embedding = nn.Embedding(vocab_size, d_model, device=device)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    d_model,
                    d_q,
                    dropout,
                    context_window,
                    batch_size,
                    mlp_layers,
                    mlp_activation,
                    n_attn_heads,
                    device,
                )
                for _ in range(n_blocks)
            ]
        )
        self.ln_f = LayerNorm(d_model, device=device)
        self.lm_head = nn.Linear(d_model, vocab_size, device=device)
        self.token_embedding.weight = self.lm_head.weight  # Weight tying

    def forward(
        self, idx: torch.Tensor, inference: bool = False
    ) -> torch.Tensor:
        t = idx.size()[-1]
        assert t <= self.context_window, (
            f"Cannot forward sequence of size {t} when context window is only {self.context_window}"
        )
        tok_emb = self.token_embedding(idx)
        x = self.dropout(tok_emb)
        for block in self.blocks:
            x = block(x)
            x = self.ln_f(x)
        if inference:
            logits = self.lm_head(x[:, -1, :])
        else:
            logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        length: int,
        temperature: float = 1.0,
        top_k: int = 5,
    ) -> torch.Tensor:
        # idx : (batch, initial_seq_length)
        for _ in range(length):
            idx_crop = (
                idx
                if idx.size(-1) <= self.context_window
                else idx[:, -self.context_window :]
            )
            logits = self.forward(idx_crop, inference=True)
            logits = logits / temperature
            assert top_k <= logits.size(-1), (
                "top k should be less than vocab size"
            )
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("Inf")
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, 1)
            idx = torch.concat((idx, idx_next), dim=1)

        return idx


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    assert x.ndim == 4
    # x: B,S,h,d
    # cos: S,d
    # sin: S,d
    x = x.transpose(1, 2)  # x: B,h,S,d
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return torch.cat([y1, y2], dim=-1).transpose(1, 2)
