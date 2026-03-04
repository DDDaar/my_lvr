import math
import torch
import torch.nn as nn
from typing import Optional


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


class Blip2QFormerTokenCompressor(nn.Module):
    """
    BLIP-2 Q-Former style compressor.
    Input: roi_tokens [N, H] or [B, N, H]
    Output: compressed tokens [B, K, H]
    """

    def __init__(
        self,
        hidden_size: int,
        num_queries: int = 8,
        num_heads: int = 8,
        num_layers: int = 1,
        dropout: float = 0.0,
        qformer_hidden_size: int = 768,
        pretrained_model_name_or_path: Optional[str] = None,
        freeze_qformer: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_queries = num_queries

        try:
            from transformers import Blip2QFormerConfig, Blip2QFormerModel
        except ImportError as exc:
            raise ImportError(
                "BLIP-2 Q-Former compressor requires transformers. "
                "Install transformers or switch --lvr_compressor_type custom."
            ) from exc

        if pretrained_model_name_or_path:
            self.qformer = self._load_qformer_from_pretrained(
                Blip2QFormerModel, pretrained_model_name_or_path
            )
            qformer_hidden_size = self.qformer.config.hidden_size
            qformer_encoder_hidden_size = getattr(
                self.qformer.config, "encoder_hidden_size", qformer_hidden_size
            )
        else:
            qformer_config = Blip2QFormerConfig(
                hidden_size=qformer_hidden_size,
                encoder_hidden_size=qformer_hidden_size,
                num_attention_heads=num_heads,
                num_hidden_layers=num_layers,
                intermediate_size=4 * qformer_hidden_size,
                hidden_dropout_prob=dropout,
                attention_probs_dropout_prob=dropout,
                cross_attention_frequency=1,
                use_qformer_text_input=False,
            )
            self.qformer = Blip2QFormerModel(qformer_config)
            qformer_encoder_hidden_size = qformer_hidden_size

        self.input_proj = (
            nn.Identity()
            if hidden_size == qformer_encoder_hidden_size
            else nn.Linear(hidden_size, qformer_encoder_hidden_size)
        )
        self.output_proj = (
            nn.Identity()
            if qformer_hidden_size == hidden_size
            else nn.Linear(qformer_hidden_size, hidden_size)
        )
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_queries, qformer_hidden_size) / math.sqrt(qformer_hidden_size)
        )

        if freeze_qformer:
            for param in self.qformer.parameters():
                param.requires_grad = False

    @staticmethod
    def _load_qformer_from_pretrained(model_cls, model_name_or_path: str):
        load_errors = []
        for kwargs in ({}, {"subfolder": "qformer"}):
            try:
                return model_cls.from_pretrained(model_name_or_path, **kwargs)
            except Exception as exc:  # pragma: no cover - runtime environment dependent
                load_errors.append(f"kwargs={kwargs}: {exc}")

        error_text = " | ".join(load_errors)
        raise RuntimeError(
            "Failed to load BLIP-2 Q-Former weights from "
            f"'{model_name_or_path}'. Details: {error_text}"
        )

    def forward(self, roi_tokens: torch.Tensor) -> torch.Tensor:
        if roi_tokens.dim() == 2:
            roi_tokens = roi_tokens.unsqueeze(0)

        batch_size = roi_tokens.size(0)
        encoder_hidden_states = self.input_proj(roi_tokens)
        encoder_attention_mask = torch.ones(
            encoder_hidden_states.shape[:2],
            dtype=torch.long,
            device=encoder_hidden_states.device,
        )

        query_tokens = self.query_tokens.expand(batch_size, -1, -1).to(encoder_hidden_states.dtype)
        outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=False,
            return_dict=True,
        )
        compressed = outputs.last_hidden_state
        return self.output_proj(compressed)


def build_lvr_token_compressor(
    compressor_type: str,
    hidden_size: int,
    num_queries: int = 8,
    num_heads: int = 8,
    num_layers: int = 1,
    dropout: float = 0.0,
    qformer_hidden_size: int = 768,
    qformer_pretrained_model_name_or_path: Optional[str] = None,
    qformer_freeze: bool = False,
) -> nn.Module:
    compressor_type = (compressor_type or "custom").lower()
    if compressor_type == "custom":
        return LVRTokenCompressor(
            hidden_size=hidden_size,
            num_queries=num_queries,
            num_heads=num_heads,
            dropout=dropout,
        )

    if compressor_type in ("qformer", "blip2_qformer", "blip-2-qformer"):
        return Blip2QFormerTokenCompressor(
            hidden_size=hidden_size,
            num_queries=num_queries,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            qformer_hidden_size=qformer_hidden_size,
            pretrained_model_name_or_path=qformer_pretrained_model_name_or_path,
            freeze_qformer=qformer_freeze,
        )

    raise ValueError(
        f"Unsupported lvr_compressor_type: '{compressor_type}'. "
        "Supported: custom, qformer."
    )
