from __future__ import annotations

import json
import time
from pathlib import Path

from diagnosis.config import load_config
from diagnosis.harness import HarnessResult, Verdict
from diagnosis.pipeline import build_ga_from_config


def _config(path: Path, output_dir: Path, workers: int) -> Path:
    data = {
        "input": {
            "requirement_file": "diagnosis/examples/AT1_AT001.py",
            "traces_file": "diagnosis/tracesAT.csv",
            "output_dir": str(output_dir),
        },
        "ga": {
            "population_size": 4,
            "generations": 0,
            "seed": 321,
            "target_sats": 10,
        },
        "evaluation": {
            "parallel_workers": workers,
        },
        "mutation": {
            "max_mutations": 1,
            "allowed_positions": [11],
            "allowed_changes": {
                "11": {
                    "numeric": [100.0, 110.0],
                },
            },
        },
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def _build_ga(config_path: Path):
    cfg = load_config(config_path)
    Path(cfg.input.output_dir).mkdir(parents=True, exist_ok=True)
    return build_ga_from_config(cfg)


def _fake_harness(filepath, timeout):
    text = Path(filepath).read_text(encoding="utf-8")
    time.sleep(0.01)
    if "<= 103." in text or "<= 104." in text or "<= 105." in text:
        return HarnessResult(Verdict.SAT, "REQUIREMENT SATISFIED\n", "", 0)
    return HarnessResult(Verdict.UNSAT, "REQUIREMENT VIOLATED\n", "", 0)


def test_parallel_subprocess_uses_per_candidate_temp_files(tmp_path, monkeypatch):
    ga = _build_ga(_config(tmp_path / "config.json", tmp_path / "out", workers=4))
    seen: list[str] = []

    def capturing_harness(filepath, timeout):
        seen.append(Path(filepath).name)
        return _fake_harness(filepath, timeout)

    monkeypatch.setattr("diagnosis.ga.run_property_script", capturing_harness)

    ga.evaluate()

    assert seen
    assert "temp.py" not in seen
    assert all(name.startswith("temp_") and name.endswith(".py") for name in seen)


def test_parallel_assignment_matches_serial(tmp_path, monkeypatch):
    serial = _build_ga(_config(tmp_path / "serial.json", tmp_path / "serial", workers=1))
    parallel = _build_ga(_config(tmp_path / "parallel.json", tmp_path / "parallel", workers=4))
    monkeypatch.setattr("diagnosis.ga.run_property_script", _fake_harness)

    serial.evaluate()
    parallel.evaluate()

    serial_rows = [
        (str(chromosome), chromosome.madeit, chromosome.fitness, chromosome.sw_score)
        for chromosome in serial.population
    ]
    parallel_rows = [
        (str(chromosome), chromosome.madeit, chromosome.fitness, chromosome.sw_score)
        for chromosome in parallel.population
    ]

    assert parallel_rows == serial_rows
    assert parallel.execution_report["parallel_workers"] == 4
