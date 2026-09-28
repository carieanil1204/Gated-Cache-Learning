"""SWE-bench(-Lite) loading, grouped into per-repo project timelines.

Per PROPOSAL.md's Workload and dataset section: each repo's issue
sequence over time stands in for a "long-running project workload."
Repos are split into disjoint pilot/main sets (see PROPOSAL.md's fix
against pilot/main-phase leakage) — a repo used in the pilot is never
reused in the main phase.

NOT YET RUN in this environment: no network access / `datasets` package
here to verify against the live HF Hub dataset. Written to the expected
schema (princeton-nlp/SWE-bench_Lite) from its documented fields —
treat as a first draft to smoke-test, not as verified-working code.
"""

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np


@dataclass
class Issue:
    repo: str
    instance_id: str
    created_at: datetime
    problem_statement: str
    patch: str  # gold patch, for the test-pass/fail oracle
    fail_to_pass: list[str] = field(default_factory=list)
    pass_to_pass: list[str] = field(default_factory=list)


@dataclass
class ProjectTimeline:
    repo: str
    issues: list[Issue]  # sorted by created_at — this IS the session sequence


def load_swebench(source: str = "swebench_lite", split: str = "test") -> list[Issue]:
    """Load SWE-bench(-Lite) and parse into Issue records."""
    from datasets import load_dataset

    hf_name = {
        "swebench_lite": "princeton-nlp/SWE-bench_Lite",
        "swebench": "princeton-nlp/SWE-bench",
    }[source]
    ds = load_dataset(hf_name, split=split)

    issues = []
    for row in ds:
        issues.append(
            Issue(
                repo=row["repo"],
                instance_id=row["instance_id"],
                created_at=datetime.fromisoformat(row["created_at"]),
                problem_statement=row["problem_statement"],
                patch=row["patch"],
                fail_to_pass=row.get("FAIL_TO_PASS", []),
                pass_to_pass=row.get("PASS_TO_PASS", []),
            )
        )
    return issues


def build_project_timelines(issues: list[Issue]) -> list[ProjectTimeline]:
    """Group issues by repo, sort each repo's issues by timestamp —
    each repo becomes one simulated project timeline (one "session
    stream"). See PROPOSAL.md's logged limitation: this aggregates
    multiple real contributors per repo, approximating rather than
    reproducing a single-user persona."""
    by_repo: dict[str, list[Issue]] = {}
    for issue in issues:
        by_repo.setdefault(issue.repo, []).append(issue)

    timelines = []
    for repo, repo_issues in by_repo.items():
        repo_issues.sort(key=lambda i: i.created_at)
        timelines.append(ProjectTimeline(repo=repo, issues=repo_issues))
    return timelines


def split_pilot_main(
    timelines: list[ProjectTimeline], pilot_fraction: float, seed: int
) -> tuple[list[ProjectTimeline], list[ProjectTimeline]]:
    """Disjoint repo-level split: a repo in the pilot set never appears
    in the main set. Fixes the effect-size-leakage risk logged in
    PROPOSAL.md — the split happens once, by repo, not by issue."""
    rng = np.random.default_rng(seed)
    n_pilot = max(1, round(len(timelines) * pilot_fraction))
    indices = rng.permutation(len(timelines))
    pilot_idx, main_idx = indices[:n_pilot], indices[n_pilot:]
    pilot = [timelines[i] for i in pilot_idx]
    main = [timelines[i] for i in main_idx]
    assert set(t.repo for t in pilot).isdisjoint(t.repo for t in main)
    return pilot, main


def resolve_via_test_oracle(issue: Issue, candidate_patch: str) -> bool | None:
    """Tier A's objective grounding for q(x,y): if the candidate answer
    is a patch, resolve correctness via SWE-bench's own FAIL_TO_PASS /
    PASS_TO_PASS test suite rather than falling back to Judge J's
    subjective score. Returns None (no oracle applicable) when the
    candidate isn't an executable patch — Judge J handles that case.

    NOT YET IMPLEMENTED: requires the SWE-bench execution harness
    (Docker-based test running) to actually apply the patch and run
    fail_to_pass/pass_to_pass — this is a stub marking where that
    integration goes, not a working implementation yet.
    """
    raise NotImplementedError(
        "Needs the SWE-bench execution harness (docker-based patch "
        "application + test run) wired in before this can resolve "
        "pass/fail. See PROPOSAL.md's Training admission section."
    )
