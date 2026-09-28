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
- Full suite: **15/15 tests passing** as of this commit.

## Not yet built

Steps 5-8 of the build order — GCL's own two ablations, Fang et al.
reimplementation, full GCL (gate/Judge/trainer), and GCL's internal
H4 ablations. Also still open: the real SWE-bench HF Hub load (confirmed
blocked here, see above) and the test-execution oracle (needs the
SWE-bench Docker harness wired in — currently a stub that raises
`NotImplementedError`); RouteLLM's `mf`/`bert`/`sw_ranking`/`causal_llm`
routers (same HF block).

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
