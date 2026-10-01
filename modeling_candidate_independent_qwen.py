"""Qwen3 decision model with isolated block-causal candidate attention."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from data import SPECIAL_TOKENS, add_decision_tokens
from modeling_candidate_independent_gemma import (
    CandidateIndependentOutput,
    build_block_causal_masks,
)


ARCHITECTURE = "candidate_independent_qwen3_block_causal_v1"


class CandidateIndependentQwen3(nn.Module):
    """Candidate-independent scorer over a Qwen3 text backbone.

    Qwen3 accepts a mapping of precomputed masks keyed by layer type. The
    Qwen3-1.7B checkpoint has full-attention layers only, but the implementation
    also handles a future Qwen3 configuration containing sliding layers.
    """

    def __init__(self, backbone: nn.Module, head_dim: int = 256):
        super().__init__()
        if getattr(backbone.config, "model_type", None) != "qwen3":
            raise ValueError("CandidateIndependentQwen3 requires a Qwen3 backbone")
        self.backbone = backbone
        self.head_dim = head_dim
        hidden_size = backbone.config.hidden_size
        self.candidate_norm = nn.LayerNorm(hidden_size)
        self.query_norm = nn.LayerNorm(hidden_size)
        self.candidate_projection = nn.Linear(hidden_size, head_dim, bias=False)
        self.query_projection = nn.Linear(hidden_size, head_dim, bias=False)
        self.score = nn.Linear(head_dim, 1, bias=False)
        self.special_token_ids: dict[str, int] = {}
        self.model_name = str(getattr(backbone.config, "_name_or_path", "unknown"))
        self.backbone.config.use_cache = False

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        tokenizer: Any,
        head_dim: int = 256,
        torch_dtype: torch.dtype | None = None,
    ) -> "CandidateIndependentQwen3":
        from transformers import AutoModel

        marker_ids = add_decision_tokens(tokenizer)
        backbone = AutoModel.from_pretrained(
            model_name,
            dtype=torch_dtype,
            attn_implementation="sdpa",
        )
        if getattr(backbone.config, "model_type", None) != "qwen3":
            raise ValueError(
                f"expected model_type='qwen3', got {backbone.config.model_type!r}"
            )
        backbone.resize_token_embeddings(len(tokenizer))
        model = cls(backbone, head_dim)
        model.model_name = model_name
        model.special_token_ids = marker_ids
        return model

    def debug_attention_masks(
        self,
        attention_mask: torch.Tensor,
        block_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        layer_types = set(self.backbone.config.layer_types)
        # The full mask does not depend on the window. Use a harmless positive
        # value while sharing the already-audited mask constructor with Gemma.
        sliding_window = getattr(self.backbone.config, "sliding_window", None)
        masks = build_block_causal_masks(
            attention_mask,
            block_ids,
            position_ids,
            int(sliding_window or self.backbone.config.max_position_embeddings),
        )
        result = {"full_attention": masks["full_attention"]}
        if "sliding_attention" in layer_types:
            if not sliding_window:
                raise ValueError("Qwen3 sliding layers require a positive sliding_window")
            result["sliding_attention"] = masks["sliding_attention"]
        unknown = layer_types - set(result)
        if unknown:
            raise ValueError(f"unsupported Qwen3 attention layer types: {sorted(unknown)}")
        return result

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        block_ids: torch.Tensor,
        position_ids: torch.Tensor,
        candidate_positions: torch.Tensor,
        candidate_mask: torch.Tensor,
        decision_positions: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_hidden: bool = False,
        **_: Any,
    ) -> CandidateIndependentOutput:
        masks = self.debug_attention_masks(attention_mask, block_ids, position_ids)
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=masks,
            position_ids=position_ids,
            use_cache=False,
        )
        hidden = outputs.last_hidden_state
        batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
        candidate = hidden.gather(
            1, candidate_positions[..., None].expand(-1, -1, hidden.shape[-1])
        )
        query = hidden[batch_indices, decision_positions]

        with torch.autocast(device_type=hidden.device.type, enabled=False):
            candidate_fp32 = self.candidate_norm(candidate.float())
            query_fp32 = self.query_norm(query.float())
            interaction = F.gelu(
                self.candidate_projection(candidate_fp32)
                + self.query_projection(query_fp32)[:, None, :]
            )
            logits = self.score(interaction).squeeze(-1)
            logits = logits.masked_fill(~candidate_mask, torch.finfo(torch.float32).min)
            log_probs = F.log_softmax(logits, dim=-1)
            loss = None
            if targets is not None:
                loss = -(targets.float() * log_probs).sum(dim=-1).mean()
        return CandidateIndependentOutput(
            logits=logits,
            log_probs=log_probs,
            loss=loss,
            candidate_hidden=candidate if return_hidden else None,
            decision_hidden=query if return_hidden else None,
        )

    def save_checkpoint(
        self, path: str | Path, tokenizer: Any, training_args: dict[str, Any]
    ) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(path / "backbone", safe_serialization=True)
        tokenizer.save_pretrained(path / "tokenizer")
        torch.save(
            {
                "candidate_norm": self.candidate_norm.state_dict(),
                "query_norm": self.query_norm.state_dict(),
                "candidate_projection": self.candidate_projection.state_dict(),
                "query_projection": self.query_projection.state_dict(),
                "score": self.score.state_dict(),
            },
            path / "decision_head.pt",
        )
        metadata = {
            "architecture": ARCHITECTURE,
            "model_name": self.model_name,
            "head_dim": self.head_dim,
            "special_token_ids": self.special_token_ids,
            "special_tokens": SPECIAL_TOKENS,
            "training_args": training_args,
        }
        (path / "decision_config.json").write_text(json.dumps(metadata, indent=2) + "\n")

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, device: str | torch.device = "cpu"
    ) -> tuple["CandidateIndependentQwen3", Any]:
        from transformers import AutoModel, AutoTokenizer

        path = Path(path)
        metadata = json.loads((path / "decision_config.json").read_text())
        if metadata.get("architecture") != ARCHITECTURE:
            raise ValueError(
                f"checkpoint architecture {metadata.get('architecture')!r} is not "
                f"{ARCHITECTURE!r}"
            )
        tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer")
        backbone = AutoModel.from_pretrained(
            path / "backbone", attn_implementation="sdpa"
        )
        model = cls(backbone, metadata["head_dim"])
        state = torch.load(path / "decision_head.pt", map_location="cpu", weights_only=True)
        for name, value in state.items():
            getattr(model, name).load_state_dict(value)
        model.model_name = metadata["model_name"]
        model.special_token_ids = metadata["special_token_ids"]
        return model.to(device), tokenizer


__all__ = ["ARCHITECTURE", "CandidateIndependentQwen3"]
