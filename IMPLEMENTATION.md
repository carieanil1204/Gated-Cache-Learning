# GCL implementation plan

Companion to `PROPOSAL.md` (the spec) — this tracks build order and
reproducibility discipline as code gets layered in. Follows the
`research-implementation` skill's checklist: seeded, config-driven,
versioned, environment-captured, logged per run, baseline verified
before proposed method.

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

## What exists so far (step 1 + a first draft of step 2)

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
  documented schema but **not yet run** in this environment (no
  network/`datasets` package available here). Treat it as an untested
  first draft until run against the real dataset.
  `resolve_via_test_oracle()` is a stub — needs the SWE-bench Docker
  execution harness wired in before it can actually resolve pass/fail.

## Not yet built

Everything in steps 3-8 above — no model code, no gate, no trainer yet.
This commit is infrastructure only, per the plan's stated scope
("scaffold the reproducibility discipline so real component code can be
layered in next").

## Tracking

MLflow (self-hosted), not wandb — Tier A is public SWE-bench data so
either would work, but MLflow avoids a swap being needed later when
Tier B (real, private user data) exists; wandb would ship that data to
an external service, violating the objective's own privacy constraint.

## How to run once step 2 finishes

```bash
pip install -r requirements.txt
python -m pytest tests/ -v          # smoke tests, no network needed
python run.py condition=cloud_only  # once run.py exists (step 3)
```
