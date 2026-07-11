from __future__ import annotations

import json
from pathlib import Path

from diagnosis.config import load_config
import pytest
from diagnosis.harness import HarnessResult, Verdict
from diagnosis.pipeline import build_ga_from_config


def _base_config(output_dir: Path) -> dict[str, object]:
    return {
        "input": {
            "requirement_file": "diagnosis/examples/AT1_AT001.py",
            "traces_file": "diagnosis/tracesAT.csv",
            "output_dir": str(output_dir),
        },
        "ga": {
            "population_size": 1,
            "generations": 0,
            "seed": 123,
            "target_sats": 2,
        },
        "mutation": {
            "max_mutations": 1,
            "allowed_positions": [11],
            "allowed_changes": {
                "11": {
                    "numeric": [100.0, 101.0],
                },
            },
        },
    }


def _write_config(path: Path, data: dict[str, object]) -> Path:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def test_default_trace_check_timeout_is_3600(tmp_path):
    cfg = load_config(_write_config(tmp_path / "config.json", _base_config(tmp_path / "out")))

    assert cfg.evaluation.trace_check_timeout_sec == 3600
    assert cfg.evaluation.cache_enabled is False
    assert cfg.evaluation.engine == "subprocess"
    assert cfg.evaluation.parallel_workers == 1


def test_custom_trace_check_timeout_reaches_harness(tmp_path, monkeypatch):
    data = _base_config(tmp_path / "out")
    data["evaluation"] = {"trace_check_timeout_sec": 17}
    cfg = load_config(_write_config(tmp_path / "config.json", data))
    Path(cfg.input.output_dir).mkdir(parents=True, exist_ok=True)
    ga = build_ga_from_config(cfg)
    captured: dict[str, int] = {}

    def fake_run_property_script(filepath, timeout):
        captured["timeout"] = timeout
        return HarnessResult(
            verdict=Verdict.SAT,
            stdout="REQUIREMENT SATISFIED\n",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr("diagnosis.ga.run_property_script", fake_run_property_script)
    monkeypatch.setattr(ga, "save_file", lambda nline: None)

    ga.evaluate()

    assert captured["timeout"] == 17


def test_worker_engine_config_is_loaded(tmp_path):
    data = _base_config(tmp_path / "out")
    data["evaluation"] = {"engine": "worker", "parallel_workers": 4}
    cfg = load_config(_write_config(tmp_path / "config.json", data))

    assert cfg.evaluation.engine == "worker"
    assert cfg.evaluation.parallel_workers == 4


def test_invalid_engine_is_rejected(tmp_path):
    data = _base_config(tmp_path / "out")
    data["evaluation"] = {"engine": "invalid"}

    with pytest.raises(Exception, match="evaluation.engine"):
        load_config(_write_config(tmp_path / "config.json", data))
