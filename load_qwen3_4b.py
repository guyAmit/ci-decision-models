"""Load the published Qwen3-4B candidate-independent checkpoint."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        default="Guy-Amit/qwen3-4b-ci-decision-4096-poc",
        help="Hugging Face repository containing the trained checkpoint",
    )
    parser.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download
    from modeling_candidate_independent_qwen import CandidateIndependentQwen3

    checkpoint = snapshot_download(repo_id=args.repo_id, repo_type="model")
    model, tokenizer = CandidateIndependentQwen3.from_checkpoint(
        checkpoint, device=args.device
    )
    model.eval()
    print(f"loaded checkpoint: {args.repo_id}")
    print(f"device: {args.device}")
    print(f"architecture: {model.__class__.__name__}")
    print(f"parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")
    print(f"vocabulary: {len(tokenizer):,}")


if __name__ == "__main__":
    main()
