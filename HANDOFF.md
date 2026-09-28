# Handoff — Gated Cache-Learning (GCL)

Read this first in a new session before doing anything else. Written
2026-09-28, end of a cloud session, to be picked up on the user's
workstation (`srmap-jc205`) once Claude credits/quota are available
there.

## Get the code

```bash
git clone https://github.com/carieanil1204/Gated-Cache-Learning.git
cd Gated-Cache-Learning
git checkout claude/charming-fermat-2imdnn
pip install -r requirements.txt
python -m pytest tests/ -v   # should show 22/22 passing
```

Branch `claude/charming-fermat-2imdnn` is **not merged into `main`** —
8 commits ahead, no PR opened for the implementation commits yet (PRs
#1 and #2, the proposal + related-work, are merged; everything since
is unmerged on this branch).

## What this project is

`PROPOSAL.md` is the full spec: Gated Cache-Learning (GCL), a
calibrated gate unifying local/cloud LLM routing, cloud-answer
write-back admission, promotion/rollback-gated adapter updates, and
post-promotion threshold recalibration. Read it — everything below
assumes familiarity with it. `IMPLEMENTATION.md` has the build order
and current build status in detail; this file is the condensed
version plus session-specific context IMPLEMENTATION.md doesn't carry
(environment findings, credit/quota state).

## Pipeline stage: 5 of 7 (implementation), steps 1-5 of 8 done

Research pipeline stages 1-4 (lit review, gap analysis, novelty check,
methodology) are closed — `PROPOSAL.md` went through 4 methodology
revision passes plus a math-validation pass (isotonic regression fix
for tau_h, objective-weight normalization, pilot/main sample
independence, etc.) after adversarial review. Don't re-litigate the
methodology; it's settled unless something concrete surfaces during
implementation.

**Stage 5 (implementation) build order, from IMPLEMENTATION.md:**
1. ✅ Reproducibility scaffolding (seeding, configs, run_log.csv, env capture)
2. ✅ SWE-bench loader — grouping/split logic tested; **real HF Hub load
   confirmed BLOCKED in the cloud sandbox this was built in** (403,
   policy denial — not applicable here on a normal network, but
   verify HF Hub reachability before assuming it works)
3. ✅ Cloud-only baseline — run, tested
4. ✅ Static router (RouteLLM) baseline — `RandomRouter` faithfully
   reproduced and verified against RouteLLM's actual source; `mf`/
   `bert`/`sw_ranking`/`causal_llm` need HF-hosted weights, same block
5. ✅ GCL's own two ablations (retrieval_no_train, local_no_gate) —
   built and verified to actually differ the way PROPOSAL.md says
6. ⬜ **NEXT** — Fang et al. (ICML 2026, arXiv:2509.24050) reimplementation.
   No public code exists (confirmed — checked arXiv/OpenReview/
   Semantic Scholar/GitHub). Needs GPU (RL post-training). This is
   where the cloud sandbox ran out of runway (no GPU there) — should
   be very doable on a workstation with one.
7. ⬜ Full GCL (Gate G, Judge J, Trainer: admission/promotion/rollback/
   recalibration) — needs a real embedding model (retrieval store so
   far uses bag-of-words/cosine similarity as a deliberate placeholder,
   see `src/models/retrieval_store.py`) and real LoRA adapter training
   (steps 3-5 used a memorization-based stand-in, not real training —
   see run.py's `run_local_no_gate`/`run_retrieval_no_train`).
8. ⬜ GCL's internal H4 ablations (feature-flags on step 7)

Run any built step: `python run.py <condition> --mock`, e.g.
`python run.py gcl_ablation_local_no_gate --mock --seed 1`.
`--mock` is currently required for everything — no real API-backed
model has been verified in any environment yet (see
`src/models/cloud_llm.py`'s `AnthropicCloudLLM`, deliberately stubbed
to fail loudly rather than silently no-op).

## Environment findings from the cloud sandbox (verify, don't assume, on a new machine)

- **No GPU**, Docker present, 4 CPU / 15GB RAM / 30GB disk, ephemeral.
- **HuggingFace Hub confirmed policy-blocked** (403 on direct curl
  check) — blocks the real SWE-bench dataset load AND RouteLLM's
  pretrained router weights. This is a sandbox-specific network
  policy, not a project design constraint — check `curl -I
  https://huggingface.co` on the new machine before assuming this
  still applies.
- **GitHub is reachable** — cloning worked fine, used to verify
  RouteLLM's actual source directly.
- **Hydra/omegaconf could not be installed** — `antlr4-python3-
  runtime==4.9.3` (omegaconf's pin) has no wheel and its sdist fails
  to build (setuptools/distutils incompatibility on Python 3.11 here).
  Worked around with `src/config.py`, a hand-rolled YAML loader with
  the same `defaults:` composition semantics — config files
  (`configs/*.yaml`) are unaffected. If Hydra installs cleanly on the
  new machine, this substitution doesn't need to be undone (it works
  fine), but you could swap back if there's a reason to.

## Credits/quota situation (why this handoff exists)

This session used a cloud-hosted "Cloud session credits" pool ($100,
scoped to cloud sessions specifically — the name is the tell). The
user's own subscription's weekly quota was at 98% (resets Thursday)
and does NOT share the credit pool — a local Claude Code session on
the workstation draws from the subscription quota, not the credits.

**Plan in effect:** write/test code in cloud sessions (using credits),
execute on the workstation with **plain `python`, no `claude` CLI
session** to avoid burning the subscription quota further. Only start
a `claude` session on the workstation once quota has reset or the
credit-scoping question is resolved (ask Anthropic support/check
billing settings for a definitive answer — this session inferred the
scoping from the label name, not from documentation).

## Two extra things this session set up — don't lose track of these

1. **Stage-3 novelty recheck scheduled**: trigger `trig_016wYKSnMSitguinuPp3Z5ZA`,
   fires 2026-10-12 into the *cloud* session that created it, not this
   one. If a new session doesn't see it fire, check `list_triggers` in
   whichever session has access, or just manually re-run a novelty
   search around then — the point is not to let the recheck lapse
   silently.
2. **Reviewer-proofing skill-choice note**: adversarial reviews so far
   used `academic-research-intelligence` (invoked directly by the user
   each time) instead of `research-reviewer-objections`, which the
   `research-pipeline` skill says is the correct one for CS/ML work.
   Not corrected retroactively — just use the right one next time if
   you're doing another review pass.

## Recommended first message to a new session here

"Continue GCL implementation from IMPLEMENTATION.md step 6 — Fang et
al. (arXiv:2509.24050) reimplementation, RL post-training. No public
code exists for it (confirmed). Read PROPOSAL.md and IMPLEMENTATION.md
first, then this handoff file for environment-specific context before
starting."
