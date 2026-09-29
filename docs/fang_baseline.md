# Fang baseline implementation

Source: [arXiv:2509.24050v4](https://arxiv.org/abs/2509.24050v4), equations
2, 5, and 19; Algorithm 1; Section 3.3.2; Tables 9–10. This is an independent
implementation, not an official reproduction or a match to reported accuracy.

The implementation samples local responses, scores successful local answers
above successful assistance, and queries the teacher at most once per prompt
group. GAPG uses `(reward - mean_reward)/(G - 1)` times the local sequence
log probability. It performs one update per fresh batch without reward standard
deviation scaling, PPO clipping, length averaging, or a cloud-token gradient.

Section 3.3.2 defines D1 by at least one correct local response. Algorithm 1's
short D1 description instead mentions positive and negative responses. We use
Section 3.3.2, retaining constant-reward groups in the count and denominator.
D2 contains no local success but a successful teacher response, capped at
`floor(rho * len(D1))`. Empty selection skips the update.

Table 10 supplies the Countdown settings. Engineering choices are SGD, optional
LoRA for the 12 GB workstation, a paraphrase of Template I, terminal marker
parsing, and sequential generation/backpropagation. Evaluation permits
`floor(N * rho/(1+rho))` cloud calls, allocated in input order. Excess requests
are regenerated locally with assistance disabled; both attempts count toward
latency. These choices must accompany any comparison with the paper.

## Install and smoke test

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements-fang.txt
.venv/bin/python -m pytest tests/ -q
.venv/bin/python run.py fang_internal_routing --mock --seed 1
```

The mock path checks routing and accounting only. It does not train a model.
Tests include an analytical gradient check and a locally initialized
Transformers model; no model download is required. Run outside the Codex
sandbox to expose CUDA devices. PCI enumeration alone does not establish CUDA
access for a process.

## Train with verified task data

`train_fang.py` runs real PyTorch updates on a Hugging Face causal model. It
accepts JSONL records with a `prompt` and a `reference_answer`. The default
Countdown verifier accepts equivalent arithmetic expressions, enforces use of
every supplied number exactly once, and evaluates using exact fractions. A row
looks like this:

```json
{"prompt":"Use 3, 3, 8, 8 exactly once to make 24.","reference_answer":{"numbers":[3,3,8,8],"target":24}}
```

The reference object is serialized to JSON before it reaches the verifier.
For other tasks, supply `--verifier your_module:verify`, a callable accepting
`(response, reference_answer)` and returning a Boolean. The optional
`src.models.fang:exact_answer` verifier supports simple exact-answer datasets.
Neither verifier grades general MATH solutions or SWE-bench patches.

Provide independently collected teacher outputs in a separate JSON file:

```json
{
  "model": "your-version-pinned-cloud-model",
  "responses": {"Your task prompt": "The teacher's actual response"}
}
```

Never fill this file with gold answers. Recorded outputs remove API spend and
teacher resampling during training, so this is a replay experiment. Training
logs distinguish logical teacher lookups from local updates. Teacher generation
cost is not measured by this entry point.

```bash
.venv/bin/python train_fang.py \
  --train-data data/train.jsonl --cloud-responses data/train_teacher.json \
  --output results/fang_seed1 --seed 1 --device cuda

.venv/bin/python eval_fang.py \
  --checkpoint results/fang_seed1/checkpoint \
  --data data/heldout.jsonl --cloud-responses data/heldout_teacher.json \
  --output results/fang_seed1_eval --device cuda
```

These paths are illustrative; no research dataset or teacher outputs are
bundled. Outputs include resolved model revision, config, seed, environment,
input hashes, training log, and checkpoint. Evaluation rejects exact training
prompt overlap and a different teacher identifier. Dataset preparation must
also enforce the proposal's disjoint repository split; text hashing alone
cannot establish repository separation. Evaluation with recorded outputs
reports local latency only, not live cloud latency or cost.

For a shorter hardware check, select a smaller causal model and override
`--steps 1 --batch-size 2`. Prompt overflow raises an error rather than silently
dropping task context. No selected prompts means no update; inspect the logged
selection counts before launching a long run.

## Remaining research validation

The SWE-bench execution oracle and live cloud client remain prerequisites for
the project's full experiment. No SWE-bench success or paper-reported accuracy
has been reproduced. Do not replace patch execution with textual gold-patch
matching. A full reproduction also needs the paper's datasets, teacher,
prompt/verifier choices, and a documented comparison of training variants.
