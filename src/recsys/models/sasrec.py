"""SASRec (Kang & McAuley, 2018): causal self-attention over the item sequence.

Inputs are left-padded to max_len with item 0. Blocks follow the common PyTorch port
(pmixer/SASRec.pytorch): pre-LayerNorm on the attention query, residual attention, LayerNorm,
point-wise FFN with residual, and a final LayerNorm. Scores = last hidden state · item embeddings.

Attention mask: causal, and real positions never attend to padding. Each position may always attend
to itself, so padding rows have a valid (ignored) softmax instead of NaN.
"""

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden), nn.Dropout(dropout), nn.ReLU(), nn.Linear(hidden, hidden), nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class SASRecBlock(nn.Module):
    def __init__(self, hidden: int, num_heads: int, dropout: float):
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden, eps=1e-8)
        self.attn = nn.MultiheadAttention(hidden, num_heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(hidden, eps=1e-8)
        self.ffn = PointWiseFeedForward(hidden, dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        q = self.attn_norm(x)
        x = q + self.attn(q, x, x, attn_mask=attn_mask, need_weights=False)[0]
        x = self.ffn(self.ffn_norm(x))
        return x * keep


class SASRec(nn.Module):
    def __init__(
        self, num_items: int, max_len: int = 50, hidden: int = 64, num_blocks: int = 2, num_heads: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_items = num_items  # item IDs 1..num_items; 0 is padding
        self.max_len = max_len
        self.num_heads = num_heads
        self.item_emb = nn.Embedding(num_items + 1, hidden, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden)
        self.emb_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(SASRecBlock(hidden, num_heads, dropout) for _ in range(num_blocks))
        self.final_norm = nn.LayerNorm(hidden, eps=1e-8)
        self._init_weights()

    def _init_weights(self) -> None:
        for name, p in self.named_parameters():
            if p.dim() > 1 and "norm" not in name:
                nn.init.xavier_normal_(p)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def _attn_mask(self, pad: torch.Tensor) -> torch.Tensor:
        """Boolean (B*heads, L, L) mask, True = blocked: future positions and padding keys (except self)."""
        L = pad.shape[1]
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=pad.device), diagonal=1)
        eye = torch.eye(L, dtype=torch.bool, device=pad.device)
        mask = causal[None] | (pad[:, None, :] & ~eye[None])
        return mask.repeat_interleave(self.num_heads, dim=0)

    def forward(self, seqs: torch.Tensor) -> torch.Tensor:
        """seqs: (B, L) left-padded item IDs, L <= max_len. Returns hidden states (B, L, hidden)."""
        L = seqs.shape[1]
        pad = seqs == 0
        keep = (~pad).unsqueeze(-1).to(self.item_emb.weight.dtype)
        positions = torch.arange(self.max_len - L, self.max_len, device=seqs.device)
        x = self.item_emb(seqs) * self.item_emb.embedding_dim**0.5 + self.pos_emb(positions)[None]
        x = self.emb_dropout(x) * keep
        attn_mask = self._attn_mask(pad)
        for block in self.blocks:
            x = block(x, attn_mask, keep)
        return self.final_norm(x)

    def item_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Dot product of hidden states (..., hidden) with all item embeddings -> (..., num_items + 1)."""
        return hidden @ self.item_emb.weight.T

    @torch.no_grad()
    def score(self, histories: Sequence[Sequence[int]]) -> torch.Tensor:
        """Scores over item IDs 0..N from the last hidden state of each (truncated, left-padded) history."""
        device = self.item_emb.weight.device
        seqs = torch.as_tensor(pad_left(histories, self.max_len), device=device)
        return self.item_logits(self.forward(seqs)[:, -1])


def pad_left(seqs: Sequence[Sequence[int]], max_len: int) -> np.ndarray:
    """Keep the last max_len items of each sequence and left-pad with 0 to shape (len(seqs), max_len)."""
    out = np.zeros((len(seqs), max_len), dtype=np.int64)
    for i, s in enumerate(seqs):
        s = s[-max_len:]
        if len(s):
            out[i, -len(s):] = s
    return out
