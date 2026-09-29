"""Evaluate a frozen Fang checkpoint against held-out data and recorded teacher outputs.

This is a quality/routing evaluation. Recorded teacher calls have zero latency;
the resulting latency measurements are not end-to-end cloud timings.
"""

import argparse
from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path

from src.eval.metrics import compute_metrics
from src.models.fang import run_fang, verified_correct
from src.models.fang_hf import HuggingFaceFangPolicy
from src.seeding import set_all_seeds
from train_fang import RecordedCloudLLM, load_examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cloud-responses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    manifest_path = args.checkpoint.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    examples = load_examples(args.data)
    training_hashes = set(manifest["training_prompt_sha256"])
    if any(hashlib.sha256(example.prompt.encode()).hexdigest() in training_hashes for example in examples):
        parser.error("evaluation data overlaps training prompts")
    cloud = RecordedCloudLLM(args.cloud_responses)
    if cloud.model != manifest["cloud_model"]:
        parser.error("evaluation teacher differs from training teacher")
    if any(example.prompt not in cloud.responses for example in examples):
        parser.error("cloud response file must cover every evaluation prompt")
    module, name = manifest["verifier"].split(":", 1)
    verify = getattr(importlib.import_module(module), name)
    args.output.mkdir(parents=True, exist_ok=False)
    set_all_seeds(manifest["seed"])
    policy = HuggingFaceFangPolicy.from_checkpoint(args.checkpoint, args.device)
    records = run_fang(manifest["config"], policy, cloud, [example.prompt for example in examples])
    for example, record in zip(examples, records):
        record["success"] = verified_correct(verify, record["response"], example.reference_answer)
    summary = {
        **asdict(compute_metrics(records)),
        "latency_scope": "local inference only; recorded teacher responses have zero latency",
        "cloud_mode": "recorded_outputs", "cloud_model": cloud.model,
        "checkpoint": str(args.checkpoint.resolve()),
        "evaluation_data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "cloud_responses_sha256": hashlib.sha256(args.cloud_responses.read_bytes()).hexdigest(),
        "budget_denied": sum(record["budget_denied"] for record in records),
    }
    (args.output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "records.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
