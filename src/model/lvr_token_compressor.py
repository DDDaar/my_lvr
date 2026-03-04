import math
import torch
import torch.nn as nn


class LVRTokenCompressor(nn.Module):
    """
    Compress variable-length ROI tokens [N, H] into fixed-length [K, H].
    """

    def __init__(
        self,
        hidden_size: int,
        num_queries: int = 8,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_queries = num_queries

        self.query = nn.Parameter(torch.randn(num_queries, hidden_size) / math.sqrt(hidden_size))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, roi_tokens: torch.Tensor) -> torch.Tensor:
        # roi_tokens: [N, H] or [B, N, H]
        if roi_tokens.dim() == 2:
            roi_tokens = roi_tokens.unsqueeze(0)

        batch_size = roi_tokens.size(0)
        query = self.query.unsqueeze(0).expand(batch_size, -1, -1)
        attn_out, _ = self.cross_attn(query, roi_tokens, roi_tokens, need_weights=False)
        hidden = self.norm1(query + attn_out)
        hidden = self.norm2(hidden + self.ffn(hidden))
        return hidden
