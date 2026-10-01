"""Serialization and batching for candidate-independent block-causal Gemma."""

from __future__ import annotations

import random
from typing import Any, Iterable

import torch

from data import JsonlDecisionDataset, add_decision_tokens, option_id, option_text, permute_example


PREFIX_BLOCK = -1
SUFFIX_BLOCK = -2
PADDING_BLOCK = -3


def _text_ids(tokenizer: Any, value: Any) -> list[int]:
    return tokenizer.encode(str(value), add_special_tokens=False)


def serialize_candidate_independent(
    row: dict[str, Any],
    tokenizer: Any,
    marker_ids: dict[str, int],
    max_length: int,
    reset_candidate_positions: bool = True,
    isolate_candidates: bool = True,
) -> dict[str, Any] | None:
    """Serialize one row with independently configurable CI mechanisms."""
    options = row["options"]
    if not options or len(options) != len(row["target"]):
        raise ValueError(f"invalid option/target lengths for {row.get('id')}")

    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    content = [
        _text_ids(tokenizer, row.get("state", "")),
        _text_ids(tokenizer, row.get("question", "")),
        *[_text_ids(tokenizer, option_text(option)) for option in options],
    ]
    fixed = len(bos) + 7 + 2 * len(options)
    if fixed > max_length:
        return None

    # Share the available content budget fairly across state, question, and
    # candidates; only ordinary text can be removed.
    allowance = max_length - fixed
    kept = [[] for _ in content]
    for depth in range(max((len(part) for part in content), default=0)):
        for index, part in enumerate(content):
            if allowance and depth < len(part):
                kept[index].append(part[depth])
                allowance -= 1
    state, question, *option_texts = kept

    # The decision token is intentionally in the shared prefix. It pools only
    # state/question, while candidates can condition on its shared query state.
    prefix = bos + [marker_ids["<state>"]] + state + [marker_ids["</state>"]]
    prefix += [marker_ids["<question>"]] + question + [marker_ids["</question>"]]
    prefix += [marker_ids["<decision>"], marker_ids["<options>"]]
    decision_position = len(prefix) - 2
    prefix_length = len(prefix)

    input_ids = list(prefix)
    block_ids = [PREFIX_BLOCK] * prefix_length
    position_ids = list(range(prefix_length))
    candidate_positions: list[int] = []
    for candidate_index, text in enumerate(option_texts):
        block = [marker_ids["<option>"]] + text + [marker_ids["</option>"]]
        input_ids.extend(block)
        block_ids.extend(([candidate_index] if isolate_candidates else [PREFIX_BLOCK]) * len(block))
        if reset_candidate_positions:
            # Every candidate begins at the same logical position, eliminating
            # serialized order from RoPE.
            position_ids.extend(range(prefix_length, prefix_length + len(block)))
        else:
            position_ids.extend(range(len(input_ids), len(input_ids) + len(block)))
        candidate_positions.append(len(input_ids) - 1)

    input_ids.append(marker_ids["</options>"])
    block_ids.append(SUFFIX_BLOCK)
    position_ids.append(prefix_length)
    assert len(input_ids) <= max_length
    return {
        "input_ids": input_ids,
        "block_ids": block_ids,
        "position_ids": position_ids,
        "candidate_positions": candidate_positions,
        "decision_position": decision_position,
        "candidate_ids": [option_id(option, row["id"], index) for index, option in enumerate(options)],
        "target": [float(value) for value in row["target"]],
        "example_id": row["id"],
    }


class CandidateIndependentCollator:
    def __init__(
        self,
        tokenizer: Any,
        max_length: int = 1024,
        permute_candidates: bool = False,
        seed: int = 42,
        reset_candidate_positions: bool = True,
        isolate_candidates: bool = True,
    ):
        self.tokenizer = tokenizer
        self.marker_ids = add_decision_tokens(tokenizer)
        self.max_length = max_length
        self.permute_candidates = permute_candidates
        self.reset_candidate_positions = reset_candidate_positions
        self.isolate_candidates = isolate_candidates
        self.rng = random.Random(seed)
        self.rejected_overlength = 0

    def __call__(self, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
        encoded = []
        for original in rows:
            row = original
            if self.permute_candidates:
                order = list(range(len(row["options"])))
                self.rng.shuffle(order)
                row = permute_example(row, order)
            item = serialize_candidate_independent(
                row,
                self.tokenizer,
                self.marker_ids,
                self.max_length,
                reset_candidate_positions=self.reset_candidate_positions,
                isolate_candidates=self.isolate_candidates,
            )
            if item is None:
                self.rejected_overlength += 1
            else:
                encoded.append(item)
        if not encoded:
            raise ValueError("all examples in this batch exceed max_length")

        batch_size = len(encoded)
        sequence_length = max(len(item["input_ids"]) for item in encoded)
        candidate_count = max(len(item["candidate_positions"]) for item in encoded)
        input_ids = torch.full(
            (batch_size, sequence_length), self.tokenizer.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros((batch_size, sequence_length), dtype=torch.long)
        block_ids = torch.full(
            (batch_size, sequence_length), PADDING_BLOCK, dtype=torch.long
        )
        position_ids = torch.zeros((batch_size, sequence_length), dtype=torch.long)
        candidate_positions = torch.zeros((batch_size, candidate_count), dtype=torch.long)
        candidate_mask = torch.zeros((batch_size, candidate_count), dtype=torch.bool)
        targets = torch.zeros((batch_size, candidate_count), dtype=torch.float32)

        for batch_index, item in enumerate(encoded):
            token_count = len(item["input_ids"])
            candidates = len(item["candidate_positions"])
            input_ids[batch_index, :token_count] = torch.tensor(item["input_ids"])
            attention_mask[batch_index, :token_count] = 1
            block_ids[batch_index, :token_count] = torch.tensor(item["block_ids"])
            position_ids[batch_index, :token_count] = torch.tensor(item["position_ids"])
            candidate_positions[batch_index, :candidates] = torch.tensor(
                item["candidate_positions"]
            )
            candidate_mask[batch_index, :candidates] = True
            targets[batch_index, :candidates] = torch.tensor(item["target"])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "block_ids": block_ids,
            "position_ids": position_ids,
            "candidate_positions": candidate_positions,
            "candidate_mask": candidate_mask,
            "decision_positions": torch.tensor(
                [item["decision_position"] for item in encoded], dtype=torch.long
            ),
            "targets": targets,
            "example_ids": [item["example_id"] for item in encoded],
            "candidate_ids": [item["candidate_ids"] for item in encoded],
        }


__all__ = [
    "CandidateIndependentCollator",
    "JsonlDecisionDataset",
    "PADDING_BLOCK",
    "PREFIX_BLOCK",
    "SUFFIX_BLOCK",
    "serialize_candidate_independent",
]
