"""Canonical JSONL loading, marker-based serialization, and dynamic batching."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset


SPECIAL_TOKENS = [
    "<state>", "</state>", "<question>", "</question>",
    "<options>", "</options>", "<option>", "</option>", "<decision>",
]


def add_decision_tokens(tokenizer: Any) -> dict[str, int]:
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    ids: dict[str, int] = {}
    for marker in SPECIAL_TOKENS:
        encoded = tokenizer.encode(marker, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"marker {marker!r} encoded as {encoded}, expected one token")
        ids[marker] = encoded[0]
    return ids


class JsonlDecisionDataset(Dataset):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        with self.path.open(encoding="utf-8") as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def permute_example(row: dict[str, Any], order: list[int]) -> dict[str, Any]:
    if sorted(order) != list(range(len(row["options"]))):
        raise ValueError("order is not a permutation of the candidate indices")
    result = dict(row)
    result["options"] = [row["options"][i] for i in order]
    result["target"] = [row["target"][i] for i in order]
    return result


def _text_ids(tokenizer: Any, value: Any) -> list[int]:
    return tokenizer.encode(str(value), add_special_tokens=False)


def option_text(option: Any) -> str:
    """Accept canonical string options and model-ready option records."""
    return str(option["text"]) if isinstance(option, dict) else str(option)


def option_id(option: Any, row_id: Any, index: int) -> str:
    """Return an option ID, synthesizing one for canonical string options."""
    if isinstance(option, dict) and "id" in option:
        return str(option["id"])
    return f"{row_id}:option:{index}"


def serialize_example(
    row: dict[str, Any], tokenizer: Any, marker_ids: dict[str, int], max_length: int
) -> dict[str, Any] | None:
    """Serialize while trimming text only; structural markers are never trimmed."""
    options = row["options"]
    if not options or len(options) != len(row["target"]):
        raise ValueError(f"invalid option/target lengths for {row.get('id')}")

    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    state = _text_ids(tokenizer, row.get("state", ""))
    question = _text_ids(tokenizer, row.get("question", ""))
    option_texts = [_text_ids(tokenizer, option_text(option)) for option in options]
    fixed = len(bos) + 7 + 2 * len(options)  # all opening/closing markers + decision
    if fixed > max_length:
        return None

    content = [state, question, *option_texts]
    allowance = max_length - fixed
    # Round-robin allocation prevents a long state from deleting every candidate's text.
    kept = [[] for _ in content]
    for depth in range(max((len(part) for part in content), default=0)):
        for index, part in enumerate(content):
            if allowance and depth < len(part):
                kept[index].append(part[depth])
                allowance -= 1

    state, question, *option_texts = kept
    sequence = bos + [marker_ids["<state>"]] + state + [marker_ids["</state>"]]
    sequence += [marker_ids["<question>"]] + question + [marker_ids["</question>"]]
    sequence += [marker_ids["<options>"]]
    candidate_positions = []
    for text in option_texts:
        sequence += [marker_ids["<option>"]] + text + [marker_ids["</option>"]]
        candidate_positions.append(len(sequence) - 1)
    sequence += [marker_ids["</options>"], marker_ids["<decision>"]]
    decision_position = len(sequence) - 1
    assert len(sequence) <= max_length
    return {
        "input_ids": sequence,
        "candidate_positions": candidate_positions,
        "decision_position": decision_position,
        "candidate_ids": [option_id(option, row["id"], index) for index, option in enumerate(options)],
        "target": [float(x) for x in row["target"]],
        "example_id": row["id"],
    }


class DecisionCollator:
    def __init__(
        self,
        tokenizer: Any,
        max_length: int = 1024,
        permute_candidates: bool = False,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.marker_ids = add_decision_tokens(tokenizer)
        self.max_length = max_length
        self.permute_candidates = permute_candidates
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
            item = serialize_example(row, self.tokenizer, self.marker_ids, self.max_length)
            if item is None:
                self.rejected_overlength += 1
            else:
                encoded.append(item)
        if not encoded:
            raise ValueError("all examples in this batch exceed max_length")

        batch_size = len(encoded)
        seq_len = max(len(item["input_ids"]) for item in encoded)
        candidates = max(len(item["candidate_positions"]) for item in encoded)
        input_ids = torch.full((batch_size, seq_len), self.tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, seq_len), dtype=torch.long)
        positions = torch.zeros((batch_size, candidates), dtype=torch.long)
        validity = torch.zeros((batch_size, candidates), dtype=torch.bool)
        targets = torch.zeros((batch_size, candidates), dtype=torch.float32)
        for batch_index, item in enumerate(encoded):
            n_tokens, n_candidates = len(item["input_ids"]), len(item["candidate_positions"])
            input_ids[batch_index, :n_tokens] = torch.tensor(item["input_ids"])
            attention_mask[batch_index, :n_tokens] = 1
            positions[batch_index, :n_candidates] = torch.tensor(item["candidate_positions"])
            validity[batch_index, :n_candidates] = True
            targets[batch_index, :n_candidates] = torch.tensor(item["target"])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "candidate_positions": positions,
            "candidate_mask": validity,
            "decision_positions": torch.tensor([x["decision_position"] for x in encoded]),
            "targets": targets,
            "example_ids": [x["example_id"] for x in encoded],
            "candidate_ids": [x["candidate_ids"] for x in encoded],
        }
