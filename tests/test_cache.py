from __future__ import annotations

import copy
import json
from pathlib import Path

from diagnosis.cache import VerdictCache, expression_key
from diagnosis.config import load_config
from diagnosis.harness import HarnessResult, Verdict
from diagnosis.pipeline import build_ga_from_config


def test_cache_round_trip(tmp_path):
    cache = VerdictCache(tmp_path / "cache.sqlite", enabled=True)
    cache.put("Not(x > 1)", "True", 0.25)

    assert expression_key("Not(x > 1)") != expression_key("Not(x > 2)")
    assert cache.get("Not(x > 1)") == ("True", 0.25)
    assert cache.stats() == {"cache_hits": 1, "cache_misses": 0, "cache_distinct": 1}


def test_cache_persists_across_instances(tmp_path):
    db_path = tmp_path / "cache.sqlite"
    cache = VerdictCache(db_path, enabled=True)
    cache.put("Not(y)", "False", 0.5)
    cache.close()

    reopened = VerdictCache(db_path, enabled=True)

    assert reopened.get("Not(y)") == ("False", 0.5)
    assert reopened.stats()["cache_distinct"] == 1


def test_disabled_cache_is_inert(tmp_path):
    cache = VerdictCache(tmp_path / "cache.sqlite", enabled=False)
    cache.put("Not(z)", "True", 1.0)

    assert cache.get("Not(z)") is None
    assert not (tmp_path / "cache.sqlite").exists()


def _config(path: Path, output_dir: Path, cache_enabled: bool) -> Path:
    data = {
        "input": {
            "requirement_file": "diagnosis/examples/AT1_AT001.py",
            "traces_file": "diagnosis/tracesAT.csv",
            "output_dir": str(output_dir),
        },
        "ga": {
            "population_size": 2,
            "generations": 0,
            "seed": 123,
            "target_sats": 2,
        },
        "evaluation": {
            "cache_enabled": cache_enabled,
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
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def test_cache_reuses_repeated_formula_verdicts(tmp_path, monkeypatch):
    cfg = load_config(_config(tmp_path / "config.json", tmp_path / "out", True))
    Path(cfg.input.output_dir).mkdir(parents=True, exist_ok=True)
    ga = build_ga_from_config(cfg)
    ga.population = [ga.population[0], copy.deepcopy(ga.population[0])]
    calls = {"count": 0}

    def fake_run_property_script(filepath, timeout):
        calls["count"] += 1
        return HarnessResult(
            verdict=Verdict.SAT,
            stdout="REQUIREMENT SATISFIED\n",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr("diagnosis.ga.run_property_script", fake_run_property_script)
    monkeypatch.setattr(ga, "save_file", lambda nline: None)

    ga.evaluate()

    assert calls["count"] == 1
    assert [chromosome.madeit for chromosome in ga.population] == ["True", "True"]
    assert ga.execution_report["cache_hits"] == 1
    assert ga.execution_report["cache_misses"] == 1
    assert ga.execution_report["cache_distinct"] == 1


def test_cache_disabled_keeps_repeated_formula_solver_calls(tmp_path, monkeypatch):
    cfg = load_config(_config(tmp_path / "config.json", tmp_path / "out", False))
    Path(cfg.input.output_dir).mkdir(parents=True, exist_ok=True)
    ga = build_ga_from_config(cfg)
    ga.population = [ga.population[0], copy.deepcopy(ga.population[0])]
    calls = {"count": 0}

    def fake_run_property_script(filepath, timeout):
        calls["count"] += 1
        return HarnessResult(
            verdict=Verdict.SAT,
            stdout="REQUIREMENT SATISFIED\n",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr("diagnosis.ga.run_property_script", fake_run_property_script)
    monkeypatch.setattr(ga, "save_file", lambda nline: None)

    ga.evaluate()

    assert calls["count"] == 2
    assert "cache_hits" not in ga.execution_report
    assert [chromosome.madeit for chromosome in ga.population] == ["True", "True"]
