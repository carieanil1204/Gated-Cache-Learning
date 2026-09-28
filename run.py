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
from src.models.local_llm import LocalLLM, MockLocalLLM
from src.models.router import RandomRouter, Router
from src.seeding import set_all_seeds

REPO_ROOT = Path(__file__).resolve().parent
RUN_LOG_PATH = REPO_ROOT / "run_log.csv"


def run_cloud_only(cfg: dict, model: CloudLLM, problem_statements: list[str]) -> list[dict]:
    records = []
    for statement in problem_statements:
        response = model.answer(statement)
        records.append(
            {
                "answered_by": "cloud",
                "latency_ms": response.latency_ms,
                "success": None,  # no test oracle wired yet — see swebench_loader.py
            }
        )
    return records


def run_static_router(
    cfg: dict,
    router: Router,
    strong_model: CloudLLM,
    weak_model: LocalLLM,
    threshold: float,
    problem_statements: list[str],
) -> list[dict]:
    """RouteLLM-style baseline (build order step 4). Each query is
    routed to strong (cloud) or weak (local) per the router's win-rate
    vs. threshold — see src/models/router.py for exactly which parts of
    RouteLLM this reproduces vs. what's blocked in this environment."""
    records = []
    for statement in problem_statements:
        decision = router.route(statement, threshold)
        if decision == "strong":
            response = strong_model.answer(statement)
            answered_by = "cloud"
        else:
            response = weak_model.answer(statement)
            answered_by = "local"
        records.append(
            {
                "answered_by": answered_by,
                "latency_ms": response.latency_ms,
                "success": None,
            }
        )
    return records


CONDITION_RUNNERS = {
    "cloud_only": run_cloud_only,
    "static_router": run_static_router,
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
    parser.add_argument("--threshold", type=float, default=0.5, help="static_router only")
    parser.add_argument("--mock", action="store_true", help="use mock models instead of real backends")
    args = parser.parse_args()

    cfg = load_config(args.condition)
    seed = args.seed if args.seed is not None else cfg["seed"]
    set_all_seeds(seed)

    if args.condition not in CONDITION_RUNNERS:
        raise NotImplementedError(
            f"condition '{args.condition}' has a config but no runner "
            "wired up yet — see IMPLEMENTATION.md's build order."
        )
    if not args.mock:
        print("No real backend verified in this environment yet — using mock models regardless of --mock. See src/models/.")

    # Placeholder problem set until the real SWE-bench load is verified
    # (src/data/swebench_loader.py's load_swebench() needs network +
    # the datasets package — huggingface.co is confirmed policy-blocked
    # in this environment, not just untested). This keeps the harness
    # runnable end-to-end now, with a clearly fake data source — not a
    # substitute for the real evaluation.
    problem_statements = [f"synthetic issue {i}" for i in range(5)]

    if args.condition == "cloud_only":
        records = run_cloud_only(cfg, MockCloudLLM(), problem_statements)
    elif args.condition == "static_router":
        records = run_static_router(
            cfg,
            router=RandomRouter(),
            strong_model=MockCloudLLM(),
            weak_model=MockLocalLLM(),
            threshold=args.threshold,
            problem_statements=problem_statements,
        )
    else:
        raise NotImplementedError(args.condition)

    metrics = compute_metrics(records)

    run_id = f"{args.condition}_seed{seed}_{int(time.time())}"
    append_run_log_row(run_id, cfg, seed, status="completed", metrics=metrics, commit=get_git_commit())

    print(f"run_id={run_id} hit_rate={metrics.hit_rate} cloud_calls={metrics.cloud_calls} "
          f"mean_latency_ms={metrics.mean_latency_ms} task_success_rate={metrics.task_success_rate}")


if __name__ == "__main__":
    main()
