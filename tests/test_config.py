from src.config import load_config


def test_cloud_only_composes_base_defaults():
    cfg = load_config("cloud_only")

    assert cfg["condition"] == "cloud_only"
    assert cfg["seed"] == 0  # from base.yaml
    assert cfg["model"]["cloud_only"] is True  # from cloud_only.yaml
    assert cfg["eval"]["significance_test"] == "wilcoxon_signed_rank"  # from base.yaml


def test_ablation_composes_through_gcl_full():
    cfg = load_config("gcl_ablation_no_admission")

    assert cfg["condition"] == "gcl_ablation_no_admission"
    assert cfg["trainer"]["admission_gate"] is False  # overridden here
    assert cfg["trainer"]["promotion_check"] is True  # inherited from gcl_full
    assert cfg["gate"]["calibration_method"] == "isotonic_regression"  # inherited
