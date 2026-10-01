"""Evaluate quality and structural invariance of candidate-independent Gemma."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import platform
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from data import option_id, permute_example
from data_candidate_independent import CandidateIndependentCollator, JsonlDecisionDataset
from modeling_candidate_independent_gemma import ARCHITECTURE, CandidateIndependentGemma


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def _progress_batches(loader, description: str):
    total = len(loader)
    started = time.perf_counter()
    for index, batch in enumerate(loader, start=1):
        yield batch
        elapsed = time.perf_counter() - started
        rate = index / elapsed if elapsed > 0 else 0.0
        remaining = (total - index) / rate if rate > 0 else 0.0
        percent = 100 * index / total if total else 100.0
        message = (
            f"{description}: {index}/{total} ({percent:5.1f}%) "
            f"elapsed {_format_duration(elapsed)}, "
            f"ETA {_format_duration(remaining)}"
        )
        print(message, end="\r", file=sys.stderr, flush=True)
    print(" " * 100, end="\r", file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--num-permutations", type=int, default=5)
    parser.add_argument("--permutation-seed", type=int, default=123)
    parser.add_argument("--invariance-tolerance", type=float, default=1e-3)
    parser.add_argument(
        "--audit-invariance",
        action="store_true",
        help="use FP32 and CUDA math SDPA for a strict numerical invariance audit",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def predict(model, tokenizer, rows: list[dict[str, Any]], args, description: str):
    collator_arguments = inspect.signature(CandidateIndependentCollator).parameters
    collator_options = {}
    if "reset_candidate_positions" in collator_arguments:
        collator_options["reset_candidate_positions"] = getattr(
            model, "reset_candidate_positions", True
        )
    if "isolate_candidates" in collator_arguments:
        collator_options["isolate_candidates"] = getattr(model, "isolate_candidates", True)
    collator = CandidateIndependentCollator(tokenizer, args.max_length, **collator_options)
    loader = DataLoader(rows, batch_size=args.batch_size, collate_fn=collator)
    probabilities, logits = {}, {}
    if str(args.device).startswith("cuda"):
        torch.cuda.synchronize()
    started = time.perf_counter()
    for batch in _progress_batches(loader, description):
        example_ids = batch.pop("example_ids")
        candidate_ids = batch.pop("candidate_ids")
        output = model(**{key: value.to(args.device) for key, value in batch.items()})
        batch_probabilities = output.log_probs.exp().cpu()
        batch_logits = output.logits.cpu()
        for index, example_id in enumerate(example_ids):
            count = len(candidate_ids[index])
            probabilities[example_id] = batch_probabilities[index, :count].tolist()
            logits[example_id] = batch_logits[index, :count].tolist()
    if str(args.device).startswith("cuda"):
        torch.cuda.synchronize()
    return probabilities, logits, time.perf_counter() - started, collator.rejected_overlength


def quality_metrics(rows, probabilities):
    correct, nll, brier, confidence, entropy = [], [], [], [], []
    by_count = defaultdict(list)
    bins = [{"count": 0, "correct": 0.0, "confidence": 0.0} for _ in range(15)]
    for row in rows:
        probs, target = probabilities[row["id"]], row["target"]
        predicted = max(range(len(probs)), key=probs.__getitem__)
        reference = max(range(len(target)), key=target.__getitem__)
        hit = float(predicted == reference)
        row_nll = -sum(t * math.log(max(p, 1e-12)) for t, p in zip(target, probs))
        correct.append(hit)
        nll.append(row_nll)
        brier.append(sum((p - t) ** 2 for p, t in zip(probs, target)))
        confidence.append(max(probs))
        entropy.append(-sum(t * math.log(t) for t in target if t > 0))
        by_count[len(target)].append((hit, row_nll))
        bin_index = min(int(max(probs) * 15), 14)
        bins[bin_index]["count"] += 1
        bins[bin_index]["correct"] += hit
        bins[bin_index]["confidence"] += max(probs)
    total = len(rows)
    grouped = {
        str(count): {
            "accuracy": sum(value[0] for value in values) / len(values),
            "nll": sum(value[1] for value in values) / len(values),
            "count": len(values),
        }
        for count, values in sorted(by_count.items())
    }
    nonbinary = [value for count, values in by_count.items() if count > 2 for value in values]
    ece = sum(
        abs(item["correct"] / item["count"] - item["confidence"] / item["count"])
        * item["count"]
        / total
        for item in bins
        if item["count"]
    )
    return {
        "count": total,
        "accuracy": sum(correct) / total,
        "nll": sum(nll) / total,
        "multiclass_brier": sum(brier) / total,
        "top_label_ece_15": ece,
        "average_confidence": sum(confidence) / total,
        "mean_target_entropy": sum(entropy) / total,
        "counts_by_candidates": dict(sorted(Counter(len(row["target"]) for row in rows).items())),
        "by_candidate_count": grouped,
        "macro_candidate_count_accuracy": sum(value["accuracy"] for value in grouped.values()) / len(grouped),
        "macro_candidate_count_nll": sum(value["nll"] for value in grouped.values()) / len(grouped),
        "nonbinary_accuracy": sum(value[0] for value in nonbinary) / len(nonbinary) if nonbinary else None,
        "nonbinary_nll": sum(value[1] for value in nonbinary) / len(nonbinary) if nonbinary else None,
    }


def js_divergence(first, second):
    midpoint = [(a + b) / 2 for a, b in zip(first, second)]
    def kl(left, right):
        return sum(x * math.log(x / y) for x, y in zip(left, right) if x > 0)
    return (kl(first, midpoint) + kl(second, midpoint)) / 2


def main(
    model_class=CandidateIndependentGemma,
    architecture: str = ARCHITECTURE,
) -> None:
    args = parse_args()
    if args.audit_invariance and str(args.device).startswith("cuda"):
        # Optimized BF16 SDPA kernels may change reduction order when candidate
        # blocks move physically. The audit path trades speed for determinism.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    model, tokenizer = model_class.from_checkpoint(args.checkpoint, args.device)
    if args.audit_invariance:
        model = model.float()
    model.eval()
    rows = JsonlDecisionDataset(args.data).rows
    original_probs, original_logits, elapsed, rejected = predict(
        model, tokenizer, rows, args, "original"
    )

    restored_prob_runs, restored_logit_runs = [], []
    permutation_accuracies = []
    position_selections, position_available = Counter(), Counter()
    for run_index in range(args.num_permutations):
        permuted_rows, displayed_orders = [], {}
        for row_index, row in enumerate(rows):
            order = list(range(len(row["options"])))
            permutation_row_index = row.get("_permutation_row_index", row_index)
            random.Random(
                f"{args.permutation_seed}:{run_index}:{permutation_row_index}:{row['id']}"
            ).shuffle(order)
            displayed_orders[row["id"]] = [option_id(row["options"][i], row["id"], i) for i in order]
            permuted_rows.append(permute_example(row, order))
        predicted_probs, predicted_logits, duration, run_rejected = predict(
            model,
            tokenizer,
            permuted_rows,
            args,
            f"permutation {run_index + 1}/{args.num_permutations}",
        )
        elapsed += duration
        rejected += run_rejected
        restored_probs, restored_logits, hits = {}, {}, []
        for row in rows:
            example_id = row["id"]
            canonical_ids = [option_id(option, row["id"], index) for index, option in enumerate(row["options"])]
            displayed_ids = displayed_orders[example_id]
            probability_map = dict(zip(displayed_ids, predicted_probs[example_id]))
            logit_map = dict(zip(displayed_ids, predicted_logits[example_id]))
            restored_probs[example_id] = [probability_map[item] for item in canonical_ids]
            restored_logits[example_id] = [logit_map[item] for item in canonical_ids]
            display_probabilities = predicted_probs[example_id]
            for position in range(len(display_probabilities)):
                position_available[position] += 1
            selected_position = max(
                range(len(display_probabilities)), key=display_probabilities.__getitem__
            )
            position_selections[selected_position] += 1
            prediction = max(
                range(len(canonical_ids)), key=restored_probs[example_id].__getitem__
            )
            reference = max(range(len(row["target"])), key=row["target"].__getitem__)
            hits.append(prediction == reference)
        restored_prob_runs.append(restored_probs)
        restored_logit_runs.append(restored_logits)
        permutation_accuracies.append(sum(hits) / len(hits))

    js_values, probability_changes, raw_logit_changes, centered_logit_changes, flips = [], [], [], [], []
    output_rows = []
    for row in rows:
        example_id = row["id"]
        base_probs, base_logits = original_probs[example_id], original_logits[example_id]
        permutation_probs = [run[example_id] for run in restored_prob_runs]
        permutation_logits = [run[example_id] for run in restored_logit_runs]
        base_selected = max(range(len(base_probs)), key=base_probs.__getitem__)
        selected = [max(range(len(values)), key=values.__getitem__) for values in permutation_probs]
        flips.append(any(value != base_selected for value in selected))
        js_values.extend(js_divergence(base_probs, values) for values in permutation_probs)
        probability_changes.extend(
            abs(a - b) for values in permutation_probs for a, b in zip(base_probs, values)
        )
        raw_logit_changes.extend(
            abs(a - b) for values in permutation_logits for a, b in zip(base_logits, values)
        )
        base_mean = sum(base_logits) / len(base_logits)
        base_centered = [value - base_mean for value in base_logits]
        for values in permutation_logits:
            run_mean = sum(values) / len(values)
            centered = [value - run_mean for value in values]
            centered_logit_changes.extend(
                abs(a - b) for a, b in zip(base_centered, centered)
            )
        candidate_ids = [option_id(option, row["id"], index) for index, option in enumerate(row["options"])]
        output_rows.append({
            "id": example_id,
            "group_id": str(row.get("group_id") or example_id),
            "source": str(row.get("source") or "unknown"),
            "candidate_ids": candidate_ids,
            "targets": row["target"],
            "original_probabilities": base_probs,
            "original_logits": base_logits,
            "permuted_probabilities": permutation_probs,
            "permuted_logits": permutation_logits,
            "selected_ids": [candidate_ids[base_selected], *[candidate_ids[i] for i in selected]],
        })

    metrics = quality_metrics(rows, original_probs)
    maximum_probability_change = max(probability_changes, default=0.0)
    maximum_raw_logit_drift = max(raw_logit_changes, default=0.0)
    maximum_centered_logit_drift = max(centered_logit_changes, default=0.0)
    metrics.update({
        "architecture": architecture,
        "examples_per_second": len(rows) * (1 + args.num_permutations) / elapsed,
        "rejected_overlength": rejected,
        "num_permutations": args.num_permutations,
        "prediction_flip_rate": sum(flips) / len(flips) if flips else 0.0,
        "mean_js_divergence": sum(js_values) / len(js_values) if js_values else 0.0,
        "max_js_divergence": max(js_values, default=0.0),
        "worst_permutation_accuracy": min(permutation_accuracies, default=metrics["accuracy"]),
        "mean_absolute_probability_change": sum(probability_changes) / len(probability_changes) if probability_changes else 0.0,
        "max_absolute_probability_change": maximum_probability_change,
        "mean_absolute_raw_logit_drift": sum(raw_logit_changes) / len(raw_logit_changes) if raw_logit_changes else 0.0,
        "max_absolute_raw_logit_drift": maximum_raw_logit_drift,
        "mean_absolute_centered_logit_drift": sum(centered_logit_changes) / len(centered_logit_changes) if centered_logit_changes else 0.0,
        "max_absolute_centered_logit_drift": maximum_centered_logit_drift,
        "invariance_tolerance": args.invariance_tolerance,
        "invariance_audit_mode": args.audit_invariance,
        "invariance_passed": (
            maximum_probability_change <= args.invariance_tolerance
            and maximum_centered_logit_drift <= args.invariance_tolerance
            if args.audit_invariance
            else None
        ),
        "position_wise_selection_frequency": {
            str(key): value / max(sum(position_selections.values()), 1)
            for key, value in sorted(position_selections.items())
        },
        "position_wise_selection_rate_when_available": {
            str(key): position_selections[key] / value
            for key, value in sorted(position_available.items())
        },
    })

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    import transformers
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "architecture": architecture,
                "versions": {
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "cuda": str(torch.version.cuda),
                    "transformers": transformers.__version__,
                },
            },
            indent=2,
        )
        + "\n"
    )
    with (output_dir / "predictions.jsonl").open("w") as handle:
        for row in output_rows:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
