"""Evaluate Qwen3 candidate-independent quality and permutation invariance."""

from evaluate_candidate_independent import main
from modeling_candidate_independent_qwen import ARCHITECTURE, CandidateIndependentQwen3


if __name__ == "__main__":
    main(model_class=CandidateIndependentQwen3, architecture=ARCHITECTURE)
