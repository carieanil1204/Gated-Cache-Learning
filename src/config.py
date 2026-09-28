"""Hand-rolled config loader — Hydra's dependency chain (omegaconf's
pinned antlr4-python3-runtime==4.9.3) fails to build in this
environment (setuptools/distutils incompatibility, not a project bug).
Same semantics as configs/*.yaml already assume: a `defaults:` list of
config names to compose first, later files overriding earlier keys.
Swap back to Hydra later if a working environment has it — config
files don't need to change, only the loader.
"""

from pathlib import Path

import yaml

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if key == "defaults":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(name: str, _seen: tuple[str, ...] = ()) -> dict:
    """Load configs/<name>.yaml, recursively composing its `defaults`
    list first (in order), then merging this file's own keys on top."""
    if name in _seen:
        raise ValueError(f"circular defaults: {_seen + (name,)}")

    path = CONFIGS_DIR / f"{name}.yaml"
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    merged: dict = {}
    for dep in raw.get("defaults", []):
        merged = _deep_merge(merged, load_config(dep, _seen + (name,)))
    merged = _deep_merge(merged, raw)
    return merged
