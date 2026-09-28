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
  p_hit (can M_L answer correctly?), a risk class, and later a quality
  score q for a candidate answer.
- **Cloud LLM M_C**: the fallback and the teacher.
- **Trainer**: admits data, fine-tunes adapters, and handles promotion
  and rollback.

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

User corrections are stored as high-weight examples.

### Update and promotion

Every N admitted pairs, or on a schedule such as overnight while
charging:

1. Train a candidate adapter A' on the buffer plus a replay sample of
   old data.
2. Evaluate A' on a held-out set and a fixed anchor set.
3. Promote A' only if quality improves without regressing on the anchor
   set. Otherwise keep the current adapter (rollback).

### Threshold recalibration

After each promotion, refit the calibration curve of p_hit against
verified outcomes and choose tau_h so that the expected local error
stays at or below a target ε:

```
tau_h = min { t : E[error | p_hit >= t] <= eps }
```

### Objective

```
minimize  a*cloud_calls + b*latency + c*error
subject to  error <= eps,  raw user data never leaves the device
```

## Evaluation plan

- **Baselines**: cloud-only, static router (RouteLLM-style), local model
  with retrieval but no training, and local training without the gate.
- **Metrics**: hit rate over time, cloud cost, latency, quality against
  cloud-only, drift and forgetting, and privacy leakage.
- **Ablations**: remove the admission gate, the promotion check, or
  recalibration, and measure which one matters most.

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

## Note on the gate model

Jev is closed and in early access, so early experiments could substitute
a small calibrated classifier for G. Jev could then be swapped in as the
gate to test its speed and calibration advantage.
