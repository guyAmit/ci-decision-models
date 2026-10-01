# CI decision models

This folder contains the small set of entry points needed to reproduce the
Qwen3-4B candidate-independent experiment on the Open-Jev dataset.

The experiment scores a variable-sized set of textual options. The
candidate-independent model gives every option the same state/question prefix,
prevents attention between option blocks, and resets option positions. The
evaluation reports decision quality and sensitivity to candidate permutations.

## Abstract

Decision models often score a variable-sized set of candidate actions encoded in
a single sequence. This setting is increasingly relevant for System 1
components inside generative systems, where candidates may be proposed or
ordered differently across runs. Standard causal cross-encoding is expressive,
but it can make a candidate's score depend on serialization order rather than
on the underlying decision problem. We introduce candidate-independent
block-causal attention, which preserves causal computation within the shared
context and each candidate while blocking cross-candidate information flow and
resetting candidate positions. We compare this architecture with standard
causal attention and complementary invariant baselines across Gemma 3 1B,
Qwen3 1.7B, and Qwen3 4B backbones. Candidate-independent attention
consistently reduces permutation sensitivity while retaining competitive
decision quality; ablations indicate that candidate isolation is the primary
source of the effect, with position resetting completing the intended symmetry.
A larger Qwen3-4B study further examines the behavior of the proposed
architecture with substantially more training data. The Qwen3-4B model
artifact is available on Hugging Face.

## Setup

This folder is self-contained. Copy it anywhere, run the commands from that
folder, install the versions in `requirements.txt`, install a CUDA-matched
PyTorch build if needed, and
authenticate with Hugging Face if the model is gated in your environment.

```bash
python -m pip install -r requirements.txt
```

## 1. Load the trained Qwen3-4B checkpoint

The project uses the trained candidate-independent checkpoint published at
[`Guy-Amit/qwen3-4b-ci-decision-4096-poc`](https://huggingface.co/Guy-Amit/qwen3-4b-ci-decision-4096-poc).
Load it directly from the Hugging Face Hub with:

```bash
python load_qwen3_4b.py --device cuda
```

The loader downloads the repository with `huggingface_hub.snapshot_download`,
then reconstructs the local `backbone/`, `tokenizer/`, and decision head saved
by this project. Use `--device cpu` when a CUDA device is unavailable. To use a
different Hugging Face repository, pass `--repo-id` explicitly. The checkpoint
can also be downloaded for evaluation, for example:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Guy-Amit/qwen3-4b-ci-decision-4096-poc",
    local_dir="checkpoints/qwen3-4b-ci-decision-4096-poc",
)
PY
```

Then set
`--checkpoint checkpoints/qwen3-4b-ci-decision-4096-poc` in the evaluation
command below.

## 2. Create Open-Jev data

The builder downloads `ZefanCai/Open-Jev`, configuration
`release-v2-redistributable`, canonicalizes the rows, removes cross-split
duplicates, and writes JSONL files plus a manifest.

```bash
python prepare_open_jev.py \
  --output-dir data/open-jev-release-v2-full \
  --train-size -1 \
  --validation-size -1 \
  --test-size -1 \
  --seed 42
```

Use `--train-size 20000 --validation-size 2000 --test-size 5000` for a small
smoke-test dataset.

## 3. Evaluate test and OOD

The checkpoint must be a trained Qwen3-4B candidate-independent checkpoint
with `decision_config.json`, `backbone/`, and `tokenizer/`.
Evaluation runs the original order and five deterministic candidate
permutations, then writes `metrics.json`, `config.json`, and
`predictions.jsonl`.

```bash
for split in test ood; do
  python evaluate_qwen3_4b.py \
    --checkpoint checkpoints/qwen3-4b-ci-decision-4096-poc \
    --data data/open-jev-release-v2-full/${split}.jsonl \
    --output-dir runs/qwen3-4b-ci/eval-${split} \
    --max-length 1024 \
    --batch-size 128
done
```

Here, “OOD” means the dataset's out-of-distribution split. The remaining Python
files in this folder are the local model, serialization, attention-mask, and
evaluation dependencies; no repository-root imports are required.
