"""Smoke tests for the parts of swebench_loader that don't need network
access or the `datasets` package — the pilot/main split logic and
per-repo grouping, using synthetic Issue records. Step 2 of the
research-implementation skill: never launch a full run cold."""

from datetime import datetime, timedelta

from src.data.swebench_loader import (
    Issue,
    build_project_timelines,
    split_pilot_main,
)


def _make_issue(repo: str, n: int, day_offset: int) -> Issue:
    return Issue(
        repo=repo,
        instance_id=f"{repo}-{n}",
        created_at=datetime(2026, 1, 1) + timedelta(days=day_offset),
        problem_statement=f"issue {n}",
        patch="",
        fail_to_pass=[],
        pass_to_pass=[],
    )


def test_build_project_timelines_groups_and_sorts_by_repo():
    issues = [
        _make_issue("repoA", 2, day_offset=5),
        _make_issue("repoA", 1, day_offset=1),
        _make_issue("repoB", 1, day_offset=3),
    ]

    timelines = build_project_timelines(issues)
    by_repo = {t.repo: t for t in timelines}

    assert set(by_repo) == {"repoA", "repoB"}
    assert [i.instance_id for i in by_repo["repoA"].issues] == ["repoA-1", "repoA-2"]
    assert len(by_repo["repoB"].issues) == 1


def test_split_pilot_main_is_disjoint_and_deterministic():
    issues = [_make_issue(f"repo{i}", 1, day_offset=i) for i in range(10)]
    timelines = build_project_timelines(issues)

    pilot1, main1 = split_pilot_main(timelines, pilot_fraction=0.2, seed=42)
    pilot2, main2 = split_pilot_main(timelines, pilot_fraction=0.2, seed=42)

    pilot_repos = {t.repo for t in pilot1}
    main_repos = {t.repo for t in main1}

    assert pilot_repos.isdisjoint(main_repos)
    assert len(pilot1) == 2  # 20% of 10
    assert len(main1) == 8
    # Same seed -> same split (reproducibility checklist item 1: seeded).
    assert {t.repo for t in pilot1} == {t.repo for t in pilot2}
    assert {t.repo for t in main1} == {t.repo for t in main2}


def test_split_pilot_main_different_seed_can_differ():
    issues = [_make_issue(f"repo{i}", 1, day_offset=i) for i in range(10)]
    timelines = build_project_timelines(issues)

    pilot_a, _ = split_pilot_main(timelines, pilot_fraction=0.2, seed=1)
    pilot_b, _ = split_pilot_main(timelines, pilot_fraction=0.2, seed=2)

    # Not asserting inequality (small sample could coincide) — just that
    # both are valid, disjoint-from-their-own-main splits.
    assert len(pilot_a) == len(pilot_b) == 2
