# Gated Cache-Learning: A Continually Trained Local Model with Calibrated Decision Gates for Cloud-LLM Pipelines

*Working title*

## Abstract

Cloud LLM assistants send every query, with its full context, to a large
remote model. This raises cost, latency, and privacy exposure, and it
discards the user's evolving problem context after each session. Hybrid
edge-cloud routing reduces some of this cost, and on-device adaptation can
personalize small models. However, routing and learning are usually
treated separately, and locally trained models risk learning their own
mistakes.

We propose Gated Cache-Learning (GCL). A small local model acts as a
semantic cache that holds compressed context for the user's ongoing
problem. A fast decision model that returns typed, calibrated outputs
(such as Jev) gates three points in the pipeline:

1. Whether the local model answers or the query escalates to the cloud.
2. Which cloud answers are trustworthy enough to become training data.
3. How the hit threshold is recalibrated as the local model improves.

The local model is updated periodically with lightweight adapters, and
each update is promoted only if it passes regression checks.

We hypothesize that GCL raises the local hit rate over time, reduces
cloud calls and latency, and keeps answer quality close to cloud-only. It
should also outperform static routing and ungated local training on
drift and error accumulation. We propose an evaluation on long-running
project workloads.

## Related work

**Edge-cloud routing.** RouteLLM (Ong et al., 2024, arXiv:2406.18665)
learns a router that picks between a weak and a strong model from
preference data, and is the de facto static-routing baseline. CoSense-LLM
(arXiv:2510.19670) goes further with PromptRouter, a cost- and
uncertainty-aware policy that chooses among edge-only, edge+RAG, or
cloud escalation, with thresholds calibrated by regret minimization on
held-out data. Zero-Shot Confidence Estimation for Small LLMs
(arXiv:2605.02241) escalates on logprob-based confidence and reports
gains over RouteLLM at matched cloud-call budgets. HybridLLM, surveyed
in Li et al.'s edge-SLM/cloud-LLM collaboration survey (arXiv:2507.16731),
predicts routing probabilities from a learned quality gap. All of these
gate a single decision — whether to call the cloud — and none feed the
resulting cloud answers back into training the local model.

**Semantic caching.** MeanCache (arXiv:2403.02694) caches cloud responses
keyed by semantic similarity to avoid repeat cloud calls, but the cache
is static: it stores past answers without ever updating the underlying
local model's weights.

**On-device continual adaptation.** CoSense-LLM's Online Personalization
stage fine-tunes local LoRA adapters on the model's own high-confidence
pseudo-labels, with elastic-penalty and block-expansion regularization
against forgetting — but this training loop is disconnected from
PromptRouter; no cloud-answer quality gate or promotion/rollback check
sits between them. Continual Learning for Sequential Personalization of
SLMs (Paula et al., 2026) monitors reference-set drift across sequential
LoRA updates but does not gate what data enters training.
The broader LoRA-continual-learning literature (O-LoRA, Wang et al. 2023;
I-LoRA, Ren et al. 2024; GainLoRA, Liang et al. 2025) targets catastrophic
forgetting during sequential fine-tuning, which GCL's anchor-set
regression check and rollback draw on, but none of these couple the
adapter update to an upstream data-admission or downstream
threshold-recalibration step.

**Unified routing and learning.** Bridging On-Device and Cloud LLMs for
Collaborative Reasoning (Fang et al., ICML 2026, arXiv:2509.24050) is the
closest attempt to unify routing and learning: it uses reinforcement-
learning post-training so the on-device model internally decides when to
offload, removing the external router entirely. This is a different
mechanism from GCL's — the routing decision is folded into the model's
own policy rather than exposed as a calibrated, auditable gate — and it
has no analogue of write-back admission, promotion/rollback safety, or
post-promotion threshold recalibration.

**Calibrated thresholds.** ConRAD (arXiv:2605.03806) derives per-operator
thresholds from a risk budget with finite-sample guarantees, in a neural
database setting; GCL's tau_h recalibration applies a similar
calibrate-to-an-error-budget idea to the local/cloud hit decision,
re-run after every adapter promotion rather than fit once.

**Positioning.** Prior work calibrates the routing decision, caches past
cloud answers, or continually adapts a local model — each in isolation.
None of the systems above put one gate in charge of routing, filtering
which cloud answers get admitted as training data, and recalibrating the
routing threshold after every promotion, with a regression-gated
promotion/rollback safety net tying the three together. That
combination, not any single piece, is GCL's contribution. This search was
not exhaustive.

## Proposed model

### Components

- **Local model M_L**: a small language model plus a retrieval store
  (facts, summaries) plus a LoRA adapter (style and problem patterns).
- **Gate G**: a calibrated decision model. For a query x it returns
  p_hit (can M_L answer correctly?) and a risk class (see Risk
  classification below).
- **Judge J**: a **third model, distinct from both M_L and M_C** —
  not the cloud teacher grading its own output. Version-pinned for the
  duration of a study so its judgments stay comparable across the run;
  never fine-tuned on GCL's own data. Scores candidate cloud answers
  for admission (see Training admission below). Breaking this out of G
  is deliberate: it keeps the admission signal from being graded by the
  same system whose training data it decides — and out of M_C, since a
  model judging its own answers as trustworthy is the same conflict of
  interest by another name.
- **Cloud LLM M_C**: the fallback and the teacher.
- **Trainer**: admits data, fine-tunes adapters, and handles promotion
  and rollback.

### Risk classification

risk(x) is one of {low, medium, high}, determined by:

- **Novelty**: embedding distance from x to the nearest retrieval-store
  entry (high distance = high novelty).
- **Safety-sensitive flag**: a fixed rule list over task type (code
  execution, financial, medical) — presence forces at least `medium`.
- **Confidence width**: the width of p_hit's confidence interval (wide
  interval = less trustworthy point estimate).

risk(x) = low only when novelty is low, no safety flag is set, and the
p_hit confidence interval is narrow. Class boundaries are calibrated
against a held-out set of labeled near-miss incidents, not hand-picked.

### Inference rule

```
if p_hit(x) >= tau_h and risk(x) = low:  answer with M_L        (hit)
else: send x + compact context to M_C                           (miss)
```

### Training admission (write-back)

A cloud pair (x, y) enters the replay buffer only if:

```
q(x, y) >= tau_q  and  user did not reject or correct y
```

**q(x, y) is scored by Judge J, not by G.** Grading admission data with
the same gate that is being trained on it is circular — G would be
scoring the data that shapes its own future behavior. J is frozen,
never updated by GCL's trainer, and is periodically spot-checked to
catch judge drift or miscalibration (a meta-calibration check on J
itself, separate from calibrating G). User rejection or correction is a
hard veto on admission regardless of q — it is the one ground-truth
signal actually available at inference time, and corrections are
stored as high-weight examples.

**On Tier A, prefer the benchmark's own oracle over J when one
exists.** SWE-bench resolves a candidate patch to objective pass/fail
against its test suite — a stronger signal than J's subjective score.
For candidates with an executable check, q(x, y) is set from that
test outcome directly; J is the fallback for the (majority of)
candidates without one. This is a Tier-A-only refinement — no such
oracle exists for general conversational queries, so Tier B and real
deployment still depend on J.

**Spot-check logistics, without violating "raw data never leaves the
device."** Outsourcing that spot-check to third-party annotators would
mean shipping private project data off-device — the same constraint
the objective already states. So the check is split by data
sensitivity: private, project-specific (x, y) pairs are spot-checked by
**the user themselves**, rating a small random sample of J's admit/
reject calls each cycle (bounded effort — on the order of 10-20 pairs
per calibration window); the general-capability anchor slice (public
benchmark data, not private) can use ordinary third-party or public
annotations, since no privacy constraint applies to it.

### Update and promotion

Every N admitted pairs, or on a schedule such as overnight while
charging:

1. Train a candidate adapter A' on the buffer plus a replay sample of
   old data.
2. Evaluate A' on a held-out set and a fixed anchor set.
3. Promote A' only if quality improves without regressing on the anchor
   set. Otherwise keep the current adapter (rollback).

**Anchor set construction and refresh.** The anchor set (~100-500
examples) is fixed at project start from two sources: recurring
project facts/tasks identified in week one, and a fixed
general-capability slice (catches forgetting outside the project
domain, not just within it). It is **append-only and versioned** —
entries are never removed, only added, every K promotions, from newly
mined hard cases — so promotion decisions stay comparable across the
project's history. A separate, deliberately *rotating* "recency"
subset sits alongside the frozen core specifically to catch drift that
an append-only anchor set would, by design, miss.

### Threshold recalibration

After each promotion, refit the calibration curve of p_hit against
verified outcomes and choose tau_h so that the expected local error
stays at or below a target ε:

```
tau_h = min { t : E[error | p_hit >= t] <= eps }
```

**Calibration metric.** "Calibrated" is reported as Expected
Calibration Error (ECE) with adaptive binning (fixed-width bins are
biased under a skewed p_hit distribution), plus Brier score and
reliability diagrams, for both p_hit and q. Small-sample risk in a
single-user setting: per-user recalibration is not trusted until at
least 50 verified outcomes near the tau_h decision boundary have
accumulated; before that threshold, tau_h falls back to a pooled
calibration curve fit across the cold-start cohort (see Workload and
dataset in the evaluation plan).

**Well-definedness of tau_h.** The `min` in the formula above is only
well-defined if E[error | p_hit >= t] is non-increasing in t — ECE and
Brier score are calibration *diagnostics*, they don't *enforce* this,
and a finite-sample empirical curve can be non-monotonic from noise
alone, especially near the small-sample boundary just described. The
calibration curve is therefore fit with **isotonic regression**
specifically (not an arbitrary binning scheme), which is monotonic by
construction and closes this gap directly. If no t in [0,1] satisfies
the constraint — even t=1 still exceeds eps — tau_h falls back to 1
(M_L never answers locally until the next promotion), rather than
leaving the min undefined.

### Objective

```
minimize  a*cloud_calls + b*latency + c*error
subject to  error <= eps,  raw user data never leaves the device
```

**Choosing a, b, c.** cloud_calls, latency, and error are on incompatible
scales (calls/day, ms, a probability), so fixed unitless weights as
written are not meaningful — this was flagged as unresolved in an
earlier review pass. Fix: normalize each term by its cloud-only
baseline value before weighting, and report a **Pareto frontier**
(sweep the weights, or the epsilon-constraint method: minimize
a*cloud_calls + b*latency at a grid of eps values) rather than
committing to one arbitrary weight vector. The eps constraint remains
a hard safety floor regardless of where on the frontier a deployment
chooses to sit; error's presence in the objective just pushes it
further down within whatever region the frontier allows.

## Evaluation plan

### Hypotheses

The abstract's claims, made falsifiable and quantified. No prior data
exists to justify a specific effect size or sample size a priori, so
H1 and H2 are run in two phases rather than asserting a threshold up
front: a **pilot phase** (small N, e.g. 5 replicate simulated users)
estimates the effect size and its variance; that estimate powers the
**main phase**'s sample size and, for H1, the growth threshold — both
reported with the pilot's estimate as justification, not picked in
advance to be achievable. **Pilot and main-phase samples are disjoint**
(no repo used in the pilot is reused in the main phase) — reusing data
across the two would leak the effect-size estimate into the
confirmatory test and inflate the apparent significance.

- **H1 (hit-rate growth)**: local hit rate increases monotonically
  from week 1 to week 4 (directional prediction fixed now); the
  specific magnitude threshold and required N are set from the pilot
  phase's effect-size estimate, tested with a Wilcoxon signed-rank test
  at p<0.05 in the main phase.
- **H2 (quality preservation)**: task-success rate vs. cloud-only
  degrades by ≤5 percentage points at matched cloud-call budget, 95%
  bootstrap CI, ≥5 seeds. (This bound is a design target from the
  objective's error constraint, not a pilot-estimated quantity — kept
  fixed.)
- **H3 (drift robustness)**: GCL's error-rate variance over the run is
  lower than ungated local training's, tested with Levene's test on
  per-window error rates.
- **H4 (component necessity)**: removing any one of {admission gate,
  promotion check, recalibration} measurably degrades H1 or H2,
  Holm-Bonferroni corrected across the ablation grid.

### Baselines

| Method | Source | Code | Reproduction risk |
|---|---|---|---|
| Cloud-only | — | trivial | none |
| Static router | RouteLLM, arXiv:2406.18665 | official repo exists, cloned and checked directly | **mixed** — the `random` router needs no weights and is faithfully reproduced (matches `RandomRouter.calculate_strong_win_rate` returning `random.uniform(0,1)` exactly, verified statistically in `tests/test_router.py`); the recommended `mf` router (and `bert`/`sw_ranking`/`causal_llm`) need pretrained weights hosted on huggingface.co/routellm, which is **confirmed policy-blocked** in the implementation environment (403, not a transient failure) — not implemented, flagged rather than silently skipped |
| Retrieval, no training | — (ablation of GCL) | internal | none |
| Local training, no gate | — (ablation of GCL) | internal | none |
| Internally-routed SLM | Fang et al., ICML 2026, arXiv:2509.24050 | **no public release found** (checked arXiv, OpenReview, Semantic Scholar, GitHub search — no repository surfaced) | **high** — reimplement from the paper's stated method (RL post-training with hierarchical rewards and group-level policy gradient) rather than assuming a faithful re-run; report the gap between reimplementation and paper-reported numbers explicitly if they diverge |

**Note on "verify against reported numbers" for the static router.**
RouteLLM's own reported numbers (85% cost reduction, 95% GPT-4
performance) are measured on MT-Bench/MMLU/GSM8K with a GPT-4/Mixtral
pair — a different benchmark entirely from SWE-bench, so a direct
number match was never going to be meaningful regardless of the HF
block. The verification that is meaningful — that the reimplemented
`random` router's behavior matches RouteLLM's own definition — is done
statistically instead (see `tests/test_router.py`).

### Workload and dataset

- **Tier A (feasible now)**: a synthetic longitudinal workload built
  from **SWE-bench** (or SWE-bench-Lite): each repository's issue
  sequence over time is naturally topic-coherent and already
  project-scoped, so per-repo issue streams stand in directly for
  "long-running project workloads" rather than needing an artificial
  clustering step. Session boundaries follow issue timestamps.
  Replicate count follows the two-phase design above — a small pilot
  (e.g. N=5 repos) sets the effect-size estimate, then the main run's N
  is sized from that pilot rather than fixed at 20 in advance. Duration
  is measured in sessions/issues, not wall clock.
- **Tier B (future work)**: an opt-in real-user pilot, gated on the
  data-retention and correction-integrity policy flagged as open in
  Systems considerations below. Tier A is what is actually run for this
  evaluation; Tier B is not claimed as complete.

### Statistics plan

Pilot phase (small N) estimates effect size and its variance for H1;
this powers the main phase's sample size, following standard practice
rather than a number picked in advance. Main phase: ≥5 seeds per
condition, mean ± 95% BCa bootstrap CI (more accurate than a plain
percentile bootstrap for ratio-type metrics like hit rate), paired
tests (Wilcoxon signed-rank) where the query stream is shared across
conditions, Holm-Bonferroni correction across the ablation family to
control the multiple-comparisons problem. The primary metric (hit rate
at fixed cloud-call budget) is pre-registered before the main phase
runs, to guard against post-hoc metric selection.

**Note on Wilcoxon's assumption.** Signed-rank is not fully
distribution-free — it assumes the paired differences are symmetric
about the median, which is plausible but not guaranteed for hit-rate
deltas across replicate repos. A sign test (weaker assumptions, less
power) is reported alongside it as a robustness check rather than
relying on Wilcoxon alone.

### Metrics

Hit rate over time, cloud cost, latency, task-success rate against
cloud-only, drift and forgetting (per-window error-rate variance),
and privacy leakage.

### Ablations

Remove the admission gate, the promotion check, or recalibration,
individually, to isolate which of the three unified gate functions
drives the H1/H2 gains — this is the direct test of H4.

## Systems considerations

The design above specifies the learning and decision logic; a production
pipeline also needs answers to the following, currently open:

- **Offline / gate-unavailable fallback.** The inference rule assumes the
  gate and cloud are always reachable. No policy is specified for what
  M_L does when M_C is unreachable or G itself is slow or down — likely
  a conservative default (answer only above a stricter tau_h, or defer).
- **Cost circuit breaker.** The objective minimizes cloud_calls in
  expectation, but nothing caps it in the worst case. A hard per-period
  budget with a corresponding fallback behavior is needed to bound
  runaway cost.
- **Drift-triggered retraining.** Promotion is currently scheduled (every
  N pairs or overnight), not triggered by detected distribution shift.
  A drift detector on the query or error distribution could trigger an
  off-cycle update instead of waiting out the schedule.
- **Staged rollout.** Promotion is binary (promote to 100% or rollback).
  A canary stage — routing a fraction of traffic to A' before full
  cutover — would catch regressions the held-out and anchor sets miss.
- **Adapter/model versioning.** Rollback implies a "current adapter" is
  tracked as a concrete, addressable artifact; this needs an explicit
  registry, not an implicit pointer.
- **Retrieval store staleness.** Facts and summaries in the retrieval
  store have no invalidation or contradiction-resolution mechanism — a
  fact cached early in a project can go stale or be contradicted by a
  later cloud answer with no path to correct it.
- **Data retention and correction integrity.** "Raw user data never
  leaves the device" is stated as a constraint but not operationalized:
  no retention/deletion policy for the replay buffer, and no defense
  against a noisy or adversarial correction poisoning the high-weight
  example set it feeds into.
- **Cold start.** Day-one M_L has an empty adapter and retrieval store,
  so p_hit is uninformative and most traffic misses by default. A
  warm-start strategy (e.g., seeding the adapter from public data before
  first use) is not addressed.
- **Observability.** No telemetry is specified for tau_h drift, hit-rate
  trend, or promotion history — needed to debug a bad promotion after
  the fact, since the gate's risk class is not otherwise explainable.

These are implementation and deployment concerns rather than gaps in the
core hypothesis, but they affect whether an evaluation on long-running
project workloads reflects a deployable system or only the idealized
gate/trainer logic above.

**Known limitations of the evaluation design, logged rather than
resolved:**

- **SWE-bench aggregates contributors, not a persona.** A repo's issue
  stream over time comes from many real people, not one continuous
  user. Treating a repo as a "simulated user" approximates the
  abstract's single-user framing; it does not reproduce it. Flagged,
  not fixed — a real fix needs either a genuinely single-author data
  source or an explicit persona-construction step not yet designed.
- **Single-rater spot-check.** The user checking Judge J's admit/
  reject calls is the only calibration signal for private data — no
  inter-rater reliability check is possible without a second person,
  which the privacy constraint rules out by construction. Accepted as
  inherent to the single-user setting, not a bug to fix.

(Objective weights a/b/c, previously logged here as unresolved, are
now fixed — see the Objective section's normalization/Pareto-frontier
fix above.)

## Note on the gate model

Jev is closed and in early access, so early experiments could substitute
a small calibrated classifier for G. Jev could then be swapped in as the
gate to test its speed and calibration advantage.
