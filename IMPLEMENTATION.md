# GCL implementation plan

**Current status source for this repository.** Updated 2026-09-29.
Read [PROPOSAL.md](PROPOSAL.md) for the research goal and evaluation design.
Read this file for implementation state and next work. [HANDOFF.md](HANDOFF.md) records the
2026-09-28 cloud-session context; its older checklist is historical.

## Current status

GCL aims to route between local and cloud language models, admit verified
cloud answers into local training, and promote adapters only after regression
checks. The local routing threshold is recalibrated after promotion. The
full method is specified in `PROPOSAL.md` but has not been implemented.

| Step | Current state | Evidence or limit |
| --- | --- | --- |
| 1. Reproducibility scaffolding | Built | Seeds, configs, run log, environment capture |
| 2. SWE-bench loader | Logic tested | Real dataset load and patch execution oracle not verified |
| 3. Cloud-only baseline | Mock run | Real cloud backend not verified |
| 4. Static router | Random baseline tested | Pretrained RouteLLM routers not loaded |
| 5. Two GCL ablations | Structural mock tests pass | Retrieval and training use stand-ins |
| 6. Fang baseline | GAPG code and GPU checks pass | No pretrained-model training or benchmark result |
| 7. Full GCL | Not built | Gate, Judge, trainer, promotion, recalibration pending |
| 8. Internal ablations | Not built | Depends on full GCL |

Validation: 55 tests passed on 2026-09-29, including CUDA gradient and small
LoRA checkpoint checks. `run_log.csv` holds smoke metrics, not research
results. `results/` contains no trained checkpoint or benchmark output yet.
Check `git status` for working-tree changes before starting new work.

**Next work:** obtain independently generated teacher responses and verified
task data for a Fang training run. Then test the real SWE-bench load and patch
oracle. Do not report mock metrics as benchmark evidence. See
[Fang baseline guide](docs/fang_baseline.md) for training inputs and commands.

## Navigation

| Need | Source |
| --- | --- |
| Research goal, method, hypotheses, evaluation | [PROPOSAL.md](PROPOSAL.md) |
| Build status, blockers, next work | This file, under Current status |
| Fang implementation and run instructions | [Fang baseline guide](docs/fang_baseline.md) |
| Cloud-session decisions and environment history | [HANDOFF.md](HANDOFF.md) |
| Experiment configuration | `configs/` |
| Smoke-run records | `run_log.csv` |
| Future checkpoints and evaluation artifacts | `results/` |

Update Current status when implementation evidence or blockers change. Keep
research rationale in [PROPOSAL.md](PROPOSAL.md) and dated session history in [HANDOFF.md](HANDOFF.md).

## Build order

Confirmed/refined from the methodology log — infra first, then easiest
to hardest baseline, full method last:

1. **Reproducibility scaffolding** (this commit) — seeding, Hydra
   configs, `run_log.csv`, env/commit capture script. Nothing runs
   without this in place.
2. **SWE-bench data loading + project-timeline construction** — must
   exist before any condition can run, since every baseline/ablation/
   proposed-method shares the same harness and data.
3. **Cloud-only baseline** — simplest possible run (every query to
   M_C). Validates the harness end-to-end with zero routing/training
   logic before anything more complex is trusted.
4. **Static router (RouteLLM)** — the actual "baseline verified before
   proposed method" checkpoint. Must be checked against RouteLLM's own
   reported numbers before any GCL-specific code is built on top.
5. **GCL's own two ablations** (retrieval-no-train, local-no-gate) —
   simpler subsets of the full system, correctly sequenced before the
   full gate assembly.
6. **Fang et al. (ICML 2026) reimplementation** — hardest baseline, no
   public code exists (confirmed absent — see `PROPOSAL.md`). Built
   after the easier baselines so the harness is already proven.
7. **Full GCL** — Gate G, Judge J, Trainer (admission/promotion/
   rollback/recalibration). Last, depends on everything above.
8. **GCL's internal ablations** (H4: remove admission gate / promotion
   check / recalibration individually) — trivial feature-flags on the
   full system once it exists; configs already scaffolded.

## Earlier implementation record: steps 1–5

The notes below describe the original cloud environment. Use Current status
above for the workstation's present state.

- `src/seeding.py` — seeds every RNG source; verified deterministic.
- `configs/` — one Hydra config per condition above, composing from
  `base.yaml`. Nothing hardcoded in run logic once run scripts exist —
  configs are the single source of hyperparameters.
- `run_log.csv` — one row per run: config, seed, status, phase,
  metric, wall-clock time, commit, timestamp, notes. Never overwritten.
- `scripts/capture_env.sh` — writes `env.txt` (pip freeze) and
  `commit.txt` (git HEAD, flags a dirty tree) into each run's output
  dir.
- `src/data/swebench_loader.py` — loads SWE-bench(-Lite), groups into
  per-repo project timelines, splits pilot/main **disjointly** (fixes
  the effect-size-leakage risk from `PROPOSAL.md`'s methodology
  passes). The grouping/split logic is tested and passing
  (`tests/test_swebench_loader.py`, 3/3 green) against synthetic data —
  the actual HF Hub load (`load_swebench()`) is written to the
  documented schema but **confirmed blocked** in this environment:
  `huggingface.co` returns 403 (policy denial via the agent proxy, not
  a transient failure — checked directly with `curl`). This is a
  limitation of *this sandbox's* network policy, not necessarily the
  final research environment (a normal workstation should reach HF
  Hub fine) — flagged here so it isn't mistaken for a design problem.
  `resolve_via_test_oracle()` is a stub — needs the SWE-bench Docker
  execution harness wired in before it can actually resolve pass/fail;
  Docker itself IS available here, so this is buildable once the data
  load happens somewhere that can reach HF.

- **Config loading**: `src/config.py` — hand-rolled YAML loader with
  `defaults:` composition, same semantics Hydra would have provided.
  **Substitution note**: Hydra/omegaconf's dependency chain
  (`antlr4-python3-runtime==4.9.3`) fails to build in this environment
  — no wheel exists for that pinned version, and the sdist build fails
  on a setuptools/distutils incompatibility unrelated to this project.
  Config *files* are unaffected (`configs/*.yaml` unchanged); only the
  loader differs. Tested against the real config files, including
  2-level inheritance (`gcl_ablation_no_admission` → `gcl_full` →
  `base`) — 2/2 passing.
- **Cloud-only baseline** (build order step 3) — `run.py`, actually run
  end-to-end: `python run.py cloud_only --mock` produces metrics and
  appends a row per metric to `run_log.csv`. Uses `MockCloudLLM` (no
  network/API key here) against a small synthetic problem set —
  **not** the real SWE-bench data yet, since that load is still
  unverified (see above). `src/models/cloud_llm.py` has the real
  backend interface (`AnthropicCloudLLM`) stubbed to fail loudly rather
  than silently no-op if used before a working API key/call exists.
- **Metrics**: `src/eval/metrics.py` — pure aggregation
  (hit_rate/cloud_calls/latency/task_success), decoupled from any
  model or data source. 3/3 tests passing.
- **Static router baseline** (build order step 4) — `run.py`, run
  end-to-end: `python run.py static_router --mock --seed 42` routes
  each query via `src/models/router.py`'s `RandomRouter` (a faithful
  reproduction of RouteLLM's actual `RandomRouter.calculate_strong_win_rate`,
  checked directly against `github.com/lm-sys/RouteLLM`'s source — GitHub
  reachable, cloned and read). RouteLLM's stronger routers (`mf`, the one
  it actually recommends, plus `bert`/`sw_ranking`/`causal_llm`) need
  pretrained weights from `huggingface.co/routellm` — same HF block as
  above, so only `random` is implemented; `UnavailableRouter` fails
  loudly if one of the others is selected rather than silently
  substituting `random`. "Verify against reported numbers" doesn't
  apply literally here — RouteLLM's numbers are on MT-Bench/MMLU/GSM8K
  with GPT-4/Mixtral, not SWE-bench — so verification is statistical
  instead: `RandomRouter`'s output distribution matches its documented
  Uniform(0,1) behavior (`tests/test_router.py`, checked over 5000
  samples).
- **GCL's own two ablations** (build order step 5) — `run.py`, both run
  end-to-end: `python run.py gcl_ablation_retrieval_no_train --mock`
  and `python run.py gcl_ablation_local_no_gate --mock`.
  `src/models/retrieval_store.py` is real (if simplified) working
  code — bag-of-words + cosine similarity, not a real sentence
  embedding model (that needs torch/transformers; deliberate scope cut
  given no GPU here — swap in a real embedding model before trusting
  retrieval-quality numbers, the logic itself is what's verified).
  The two ablations are built to actually differ the way
  `PROPOSAL.md` describes them, and this is tested, not just claimed:
  `retrieval_no_train`'s store never grows (0% hit rate on 3
  near-duplicate queries, all escalate); `local_no_gate`'s store
  absorbs every cloud answer with zero filtering (queries 2-3 hit
  locally after query 1's escalation gets memorized) —
  `tests/test_gcl_ablations.py`, 3/3 green. Real LoRA training is
  stood in for by this memorization mechanism throughout, since actual
  adapter training needs a GPU this environment doesn't have — the
  *structural* behavior (unconditional vs. no write-back) is exercised
  here; measuring the *harm* of unconditional write-back (H3) needs
  the test oracle, still blocked.
- Full suite: **22/22 tests passing** as of this commit.

## Step 6 implementation (workstation, 2026-09-29)

The Fang baseline now has an independent GAPG implementation based on
arXiv:2509.24050v4. `src/models/fang.py` handles verified hierarchical rewards,
one teacher call per prompt group, prompt filtering, and budgeted inference.
`src/models/fang_hf.py` provides Hugging Face rollouts, sequence likelihoods,
SGD updates, optional LoRA, and checkpoint saving/loading. `train_fang.py`
trains from explicit task data and recorded teacher outputs; `eval_fang.py`
checks held-out quality/routing with a frozen checkpoint. The existing harness
also supports `python run.py fang_internal_routing --mock`.

See `docs/fang_baseline.md` for commands and implementation choices. These
include resolving the paper's differing D1 descriptions in favor of Section
3.3.2, a paraphrased prompt, optional LoRA, and recorded teacher outputs.
These choices are not evidence of reproducing the paper's reported accuracy.
The default Countdown verifier checks arithmetic and number usage; SWE-bench
still requires an execution oracle.

The workstation has an RTX 3060 with 12,288 MiB VRAM. Both NVIDIA device
access and network access require execution outside the Codex sandbox:
`nvidia-smi` succeeds there, and `curl -I https://huggingface.co` returns 200.
The previous sandbox's no-GPU/HF-policy-block findings do not describe this
host. A project-local `.venv` is used for dependencies.

Verification: **55 tests passed**, including CUDA gradient checks and a real
Transformers/PEFT LoRA update and checkpoint round trip on the RTX 3060.
The test model is a tiny randomly initialized Qwen2, not pretrained Qwen2.5
or a research result. The mock Fang CLI completed and appended its smoke
metrics to `run_log.csv`; `pip check` found no broken dependencies.

## Not yet built or validated

The Fang baseline's full training run and benchmark reproduction remain
unvalidated. Steps 7-8 of the build order — full GCL
(gate/Judge/trainer, needs a real embedding model + real adapter
training, not the retrieval-store stand-in used for the ablations), and
GCL's internal H4 ablations. Also still open: the real SWE-bench HF Hub
dataset load (host reachability is now confirmed) and the test-execution oracle (needs the
SWE-bench Docker harness wired in); RouteLLM's
`mf`/`bert`/`sw_ranking`/`causal_llm` routers (weights not yet loaded here).

## Tracking

MLflow (self-hosted), not wandb — Tier A is public SWE-bench data so
either would work, but MLflow avoids a swap being needed later when
Tier B (real, private user data) exists; wandb would ship that data to
an external service, violating the objective's own privacy constraint.

## Current quick checks

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python run.py cloud_only --mock
.venv/bin/python run.py fang_internal_routing --mock
```

The mock commands append smoke records to `run_log.csv`. Real Fang training
and evaluation use `train_fang.py` and `eval_fang.py` with verified input data;
see `docs/fang_baseline.md`.
