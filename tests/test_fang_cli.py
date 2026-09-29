import hashlib
import json
import sys

import pytest

from train_fang import RecordedCloudLLM, load_examples


def test_duplicate_training_prompts_are_rejected(tmp_path):
    path = tmp_path / "data.jsonl"
    row = json.dumps({"prompt": "q", "reference_answer": "a"}) + "\n"
    path.write_text(row * 2)
    with pytest.raises(ValueError, match="duplicate"):
        load_examples(path)


def test_recorded_teacher_never_substitutes_a_missing_answer(tmp_path):
    path = tmp_path / "teacher.json"
    path.write_text(json.dumps({"model": "teacher-version", "responses": {"q": "wrong"}}))
    cloud = RecordedCloudLLM(path)
    assert cloud.answer("q").text == "wrong"
    with pytest.raises(ValueError, match="missing"):
        cloud.answer("other")
    assert cloud.call_count == 1


def test_evaluation_rejects_training_overlap_before_model_load(tmp_path, monkeypatch):
    import eval_fang

    (tmp_path / "manifest.json").write_text(json.dumps({
        "training_prompt_sha256": [hashlib.sha256(b"q").hexdigest()],
    }))
    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps({"prompt": "q", "reference_answer": "a"}) + "\n")
    monkeypatch.setattr(sys, "argv", [
        "eval_fang.py", "--checkpoint", str(tmp_path / "checkpoint"),
        "--data", str(data), "--cloud-responses", "unused.json",
        "--output", str(tmp_path / "output"),
    ])
    with pytest.raises(SystemExit) as error:
        eval_fang.main()
    assert error.value.code == 2
    assert not (tmp_path / "output").exists()


def test_harness_requires_explicit_mock_flag(tmp_path, monkeypatch):
    import run

    monkeypatch.setattr(sys, "argv", ["run.py", "fang_internal_routing"])
    monkeypatch.setattr(run, "RUN_LOG_PATH", tmp_path / "run_log.csv")
    with pytest.raises(SystemExit) as error:
        run.main()
    assert error.value.code == 2
    assert not run.RUN_LOG_PATH.exists()
