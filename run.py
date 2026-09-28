"""Run entry point. Usage: python run.py <condition_name> [--seed N]

Implements the cloud-only condition end-to-end (build order step 3).
Other conditions (static_router, gcl_full, etc.) will raise
NotImplementedError until their steps are built — this deliberately
doesn't silently no-op on an unbuilt condition.
"""

import argparse
import csv
import time
from datetime import datetime, timezone
from pathlib import Path

from src.config import load_config
from src.eval.metrics import compute_metrics
from src.models.cloud_llm import CloudLLM, MockCloudLLM
from src.seeding import set_all_seeds

REPO_ROOT = Path(__file__).resolve().parent
RUN_LOG_PATH = REPO_ROOT / "run_log.csv"


def run_cloud_only(cfg: dict, model: CloudLLM, problem_statements: list[str]) -> list[dict]:
    records = []
    for statement in problem_statements:
        start = time.monotonic()
        response = model.answer(statement)
        elapsed_ms = (time.monotonic() - start) * 1000
        records.append(
            {
                "answered_by": "cloud",
                # Prefer the model's own reported latency (meaningful for a
                # mock/deterministic backend); wall-clock elapsed_ms is
                # available too but not used here to avoid double-counting.
                "latency_ms": response.latency_ms,
                "success": None,  # no test oracle wired yet — see swebench_loader.py
            }
        )
    return records


CONDITION_RUNNERS = {
    "cloud_only": run_cloud_only,
}


def append_run_log_row(run_id: str, cfg: dict, seed: int, status: str, metrics, commit: str) -> None:
    is_new = not RUN_LOG_PATH.exists()
    with open(RUN_LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(
                [
                    "run_id",
                    "config",
                    "seed",
                    "status",
                    "phase",
                    "metric_name",
                    "metric_value",
                    "wall_clock_s",
                    "commit",
                    "timestamp",
                    "notes",
                ]
            )
        timestamp = datetime.now(timezone.utc).isoformat()
        for name, value in [
            ("hit_rate", metrics.hit_rate),
            ("cloud_calls", metrics.cloud_calls),
            ("mean_latency_ms", metrics.mean_latency_ms),
            ("task_success_rate", metrics.task_success_rate),
        ]:
            writer.writerow([run_id, cfg["condition"], seed, status, "smoke", name, value, "", commit, timestamp, ""])


def get_git_commit() -> str:
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("condition")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mock", action="store_true", help="use MockCloudLLM instead of a real backend")
    args = parser.parse_args()

    cfg = load_config(args.condition)
    seed = args.seed if args.seed is not None else cfg["seed"]
    set_all_seeds(seed)

    runner = CONDITION_RUNNERS.get(args.condition)
    if runner is None:
        raise NotImplementedError(
            f"condition '{args.condition}' has a config but no runner "
            "wired up yet — see IMPLEMENTATION.md's build order."
        )

    model = MockCloudLLM()
    if not args.mock:
        print("No real CloudLLM backend verified in this environment yet — using MockCloudLLM regardless of --mock. See src/models/cloud_llm.py.")

    # Placeholder problem set until the real SWE-bench load is verified
    # (src/data/swebench_loader.py's load_swebench() needs network +
    # the datasets package, not available in this environment). This
    # keeps the harness runnable end-to-end now, with a clearly fake
    # data source — not a substitute for the real evaluation.
    problem_statements = [f"synthetic issue {i}" for i in range(5)]

    records = runner(cfg, model, problem_statements)
    metrics = compute_metrics(records)

    run_id = f"{args.condition}_seed{seed}_{int(time.time())}"
    append_run_log_row(run_id, cfg, seed, status="completed", metrics=metrics, commit=get_git_commit())

    print(f"run_id={run_id} hit_rate={metrics.hit_rate} cloud_calls={metrics.cloud_calls} "
          f"mean_latency_ms={metrics.mean_latency_ms} task_success_rate={metrics.task_success_rate}")


if __name__ == "__main__":
    main()
