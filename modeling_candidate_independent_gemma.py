"""Gemma decision model with isolated block-causal candidate attention."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from data import SPECIAL_TOKENS, add_decision_tokens
from data_candidate_independent import PADDING_BLOCK, PREFIX_BLOCK, SUFFIX_BLOCK


ARCHITECTURE = "candidate_independent_block_causal_v1"


@dataclass
class CandidateIndependentOutput:
    logits: torch.Tensor
    log_probs: torch.Tensor
    loss: torch.Tensor | None = None
    candidate_hidden: torch.Tensor | None = None
    decision_hidden: torch.Tensor | None = None


def build_block_causal_masks(
    attention_mask: torch.Tensor,
    block_ids: torch.Tensor,
    position_ids: torch.Tensor,
    sliding_window: int,
) -> dict[str, torch.Tensor]:
    """Build boolean SDPA masks shaped [batch, 1, query, key]."""
    if sliding_window <= 0:
        raise ValueError("sliding_window must be positive")
    if not (attention_mask.shape == block_ids.shape == position_ids.shape):
        raise ValueError("attention_mask, block_ids, and position_ids must have equal shapes")

    query_block = block_ids[:, :, None]
    key_block = block_ids[:, None, :]
    query_position = position_ids[:, :, None]
    key_position = position_ids[:, None, :]

    query_is_prefix = query_block == PREFIX_BLOCK
    query_is_candidate = query_block >= 0
    query_is_suffix = query_block == SUFFIX_BLOCK
    key_is_prefix = key_block == PREFIX_BLOCK

    prefix_edges = query_is_prefix & key_is_prefix & (key_position <= query_position)
    candidate_edges = query_is_candidate & (
        key_is_prefix
        | ((key_block == query_block) & (key_position <= query_position))
    )
    suffix_edges = query_is_suffix & (
        key_is_prefix
        | ((key_block == SUFFIX_BLOCK) & (key_position <= query_position))
    )
    allowed = prefix_edges | candidate_edges | suffix_edges

    valid_query = attention_mask[:, :, None].bool()
    valid_key = attention_mask[:, None, :].bool()
    not_padding_blocks = (query_block != PADDING_BLOCK) & (key_block != PADDING_BLOCK)
    full = allowed & valid_query & valid_key & not_padding_blocks

    distance = query_position - key_position
    local = full & (distance >= 0) & (distance < sliding_window)
    return {
        "full_attention": full[:, None, :, :],
        "sliding_attention": local[:, None, :, :],
    }


def build_causal_masks(
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    sliding_window: int,
) -> dict[str, torch.Tensor]:
    """Build ordinary sequence-causal masks, optionally with custom RoPE IDs."""
    if sliding_window <= 0:
        raise ValueError("sliding_window must be positive")
    batch_size, sequence_length = attention_mask.shape
    sequence_positions = torch.arange(
        sequence_length, device=attention_mask.device, dtype=torch.long
    )
    query_position = sequence_positions[None, :, None]
    key_position = sequence_positions[None, None, :]
    allowed = key_position <= query_position
    valid_query = attention_mask[:, :, None].bool()
    valid_key = attention_mask[:, None, :].bool()
    full = allowed & valid_query & valid_key
    distance = query_position - key_position
    local = full & (distance >= 0) & (distance < sliding_window)
    return {
        "full_attention": full[:, None, :, :],
        "sliding_attention": local[:, None, :, :],
    }


class CandidateIndependentGemma(nn.Module):
    def __init__(self, backbone: nn.Module, head_dim: int = 256):
        super().__init__()
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
        self.backbone.config.use_bidirectional_attention = False
        self.isolate_candidates = True
        self.reset_candidate_positions = True

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        tokenizer: Any,
        head_dim: int = 256,
        torch_dtype: torch.dtype | None = None,
    ) -> "CandidateIndependentGemma":
        from transformers import AutoModel

        marker_ids = add_decision_tokens(tokenizer)
        backbone = AutoModel.from_pretrained(
            model_name,
            dtype=torch_dtype,
            attn_implementation="sdpa",
        )
        backbone.resize_token_embeddings(len(tokenizer))
        backbone.config.use_bidirectional_attention = False
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
        if self.isolate_candidates:
            return build_block_causal_masks(
                attention_mask,
                block_ids,
                position_ids,
                int(self.backbone.config.sliding_window),
            )
        return build_causal_masks(
            attention_mask, position_ids, int(self.backbone.config.sliding_window)
        )

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
        self.backbone.config.use_bidirectional_attention = False
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
    ) -> tuple["CandidateIndependentGemma", Any]:
        from transformers import AutoModel, AutoTokenizer

        path = Path(path)
        metadata = json.loads((path / "decision_config.json").read_text())
        if metadata.get("architecture") != ARCHITECTURE:
            raise ValueError(
                f"checkpoint architecture {metadata.get('architecture')!r} is not {ARCHITECTURE!r}"
            )
        tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer")
        backbone = AutoModel.from_pretrained(path / "backbone", attn_implementation="sdpa")
        backbone.config.use_bidirectional_attention = False
        model = cls(backbone, metadata["head_dim"])
        state = torch.load(path / "decision_head.pt", map_location="cpu", weights_only=True)
        for name, value in state.items():
            getattr(model, name).load_state_dict(value)
        model.model_name = metadata["model_name"]
        model.special_token_ids = metadata["special_token_ids"]
        training_args = metadata.get("training_args", {})
        model.isolate_candidates = training_args.get("isolate_candidates", True)
        model.reset_candidate_positions = training_args.get("reset_candidate_positions", True)
        return model.to(device), tokenizer
