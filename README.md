# Gated Cache-Learning

Gated Cache-Learning (GCL) is a research prototype for a local language model
that learns from selected cloud answers. A calibrated gate decides when to use
the local model, which cloud answers may enter training, and when to adjust the
local routing threshold after an adapter update. The full research design is
in [PROPOSAL.md](PROPOSAL.md).

## Start here

| Need | Read |
| --- | --- |
| Goal, method, hypotheses, evaluation plan | [PROPOSAL.md](PROPOSAL.md) |
| Current progress, blockers, next work | [IMPLEMENTATION.md](IMPLEMENTATION.md#current-status) |
| Fang baseline training and evaluation | [docs/fang_baseline.md](docs/fang_baseline.md) |
| Earlier session decisions and environment history | [HANDOFF.md](HANDOFF.md) |

`IMPLEMENTATION.md` owns current status. `HANDOFF.md` is dated history; its
original checklist describes an earlier cloud session.

## Check the code

Use Python 3.11 or newer. Create a local environment, install the baseline
dependencies, and run the test suite:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements-fang.txt
.venv/bin/python -m pytest tests/ -q
```

Run a mock routing check without model downloads or cloud credentials:

```bash
.venv/bin/python run.py fang_internal_routing --mock --seed 1
```

Mock metrics go to [run_log.csv](run_log.csv). Trained checkpoints and
evaluation artifacts go under `results/` when real runs occur. Mock metrics
are not research benchmark results. For real Fang training inputs and commands,
read [docs/fang_baseline.md](docs/fang_baseline.md).

Update [IMPLEMENTATION.md](IMPLEMENTATION.md#current-status) when a milestone,
blocker, or validation result changes. Keep research decisions in
[PROPOSAL.md](PROPOSAL.md) and dated session context in [HANDOFF.md](HANDOFF.md).
