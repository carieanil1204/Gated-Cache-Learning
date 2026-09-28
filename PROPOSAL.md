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

## Note on the gate model

Jev is closed and in early access, so early experiments could substitute
a small calibrated classifier for G. Jev could then be swapped in as the
gate to test its speed and calibration advantage.
