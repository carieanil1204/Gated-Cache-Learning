"""Offline GAPG training using explicitly supplied data and cloud responses.

This entry point supports Countdown and custom verifiers. SWE-bench requires a patch
execution verifier; pass a custom module:function verifier when it exists.
Cloud responses must come from the chosen teacher, never the gold answers.
"""

import argparse
from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path
import random
import subprocess
import sys

from src.config import load_config
from src.models.cloud_llm import CloudLLM, CloudResponse
from src.models.fang import FangExample, train_step
from src.models.fang_hf import HuggingFaceFangPolicy
from src.seeding import set_all_seeds


class RecordedCloudLLM(CloudLLM):
    """Replay an identified teacher's answers; no API spend or gold substitution."""

    def __init__(self, path: Path):
        content = json.loads(path.read_text())
        if not isinstance(content.get("model"), str) or not content["model"].strip():
            raise ValueError("cloud response file requires a nonempty model identifier")
        self.model = content["model"]
        self.responses = content["responses"]
        if not isinstance(self.responses, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in self.responses.items()
        ):
            raise ValueError("cloud responses must map prompt strings to response strings")
        self.call_count = 0

    def answer(self, problem_statement: str) -> CloudResponse:
        if problem_statement not in self.responses:
            raise ValueError("missing recorded teacher response for prompt")
        self.call_count += 1
        # Lookup time is not a claim about actual cloud inference latency.
        return CloudResponse(self.responses[problem_statement], 0.0)


def load_examples(path: Path) -> list[FangExample]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("training data is empty")
    examples = []
    seen = set()
    for row in rows:
        prompt, reference = row["prompt"], row["reference_answer"]
        if isinstance(reference, dict):
            reference = json.dumps(reference, sort_keys=True)
        if not isinstance(prompt, str) or not prompt.strip() or not isinstance(reference, str) or not reference.strip():
            raise ValueError("prompt and reference_answer must be nonempty strings")
        if prompt in seen:
            raise ValueError("duplicate training prompt")
        seen.add(prompt)
        examples.append(FangExample(prompt, reference))
    return examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True, help="JSONL: prompt, reference_answer")
    parser.add_argument("--cloud-responses", type=Path, required=True, help="JSON: model, responses (prompt to text)")
    parser.add_argument("--output", type=Path, required=True, help="new run directory")
    parser.add_argument("--config", default="fang_internal_routing")
    parser.add_argument("--verifier", help="module:function; defaults to the config's verifier")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--model")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    settings = cfg["fang"]
    seed = args.seed if args.seed is not None else cfg["seed"]
    if args.model is not None:
        settings["model_name"] = args.model
        settings["revision"] = None
    if args.steps is not None:
        settings["training_steps"] = args.steps
    if args.batch_size is not None:
        settings["batch_size"] = args.batch_size
    if settings["training_steps"] < 1 or settings["batch_size"] < 1:
        parser.error("steps and batch size must be positive")
    examples = load_examples(args.train_data)
    cloud = RecordedCloudLLM(args.cloud_responses)
    if any(example.prompt not in cloud.responses for example in examples):
        parser.error("cloud response file must cover every training prompt")
    verifier_name = args.verifier or settings["verifier"]
    module, name = verifier_name.split(":", 1)
    verify = getattr(importlib.import_module(module), name)
    if not callable(verify):
        parser.error("verifier must be callable")
    args.output.mkdir(parents=True, exist_ok=False)
    set_all_seeds(seed)
    rng = random.Random(seed)
    policy = HuggingFaceFangPolicy.from_pretrained(settings, args.device)
    # Fail before the first rollout if an example would be silently truncated.
    for example in examples:
        policy._prompt_ids(example.prompt)
    root = Path(__file__).resolve().parent
    manifest = {
        "config": cfg, "seed": seed, "resolved_model": policy.metadata,
        "cloud_model": cloud.model, "cloud_mode": "recorded_outputs",
        "verifier": verifier_name, "device": args.device,
        "training_data_sha256": hashlib.sha256(args.train_data.read_bytes()).hexdigest(),
        "training_prompt_sha256": [hashlib.sha256(example.prompt.encode()).hexdigest() for example in examples],
        "cloud_responses_sha256": hashlib.sha256(args.cloud_responses.read_bytes()).hexdigest(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "dirty_tree": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "env.txt").write_text(subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True,
    ))
    with (args.output / "training.jsonl").open("w") as log:
        for step in range(settings["training_steps"]):
            batch = rng.sample(examples, min(settings["batch_size"], len(examples)))
            outcome = train_step(
                batch, policy, cloud, verify,
                settings["group_size"], settings["rho"],
                settings["accuracy_reward"], settings["coordination_reward"], rng,
            )
            row = {"step": step, **asdict(outcome)}
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps(row), flush=True)
    policy.save(args.output / "checkpoint")


if __name__ == "__main__":
    main()
