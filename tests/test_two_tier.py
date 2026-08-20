"""Two-tier timeout tests (Feature 3): z3-timeout injection + escalation shape.

The controller-level escalation and region-memory decisions are covered in
test_inference.py; here we pin the subprocess-engine z3-timeout injection and
the config interaction rule.
"""

from __future__ import annotations

import pytest

from diagnosis.config import ConfigError, load_config
from diagnosis.ga import replace_property_assertion


BASE = [
    "def AT1():\n",
    "\tz3solver=Solver()\n",
    "\tz3solver.add(Not(ForAll([t], phi)))\n",
    "\tstatus=z3solver.check()\n",
]


def test_no_injection_is_byte_identical():
    out = replace_property_assertion(BASE, "Not(ForAll([t], psi))")
    # exactly one line replaced, no set("timeout") line added
    assert sum(1 for l in out if "set(\"timeout\"" in l) == 0
    assert out[2] == "\tz3solver.add(Not(ForAll([t], psi)))\n"
    assert len(out) == len(BASE)


def test_injection_adds_timeout_line_before_assertion():
    out = replace_property_assertion(BASE, "Not(ForAll([t], psi))", z3_timeout_ms=60000)
    # one extra logical line, preserving indentation
    assert "\tz3solver.set(\"timeout\", 60000)\n" in "".join(out)
    joined = "".join(out)
    # the timeout is set before the add
    assert joined.index("set(\"timeout\"") < joined.index("z3solver.add(")


def test_config_rejects_high_tier_above_trace_timeout(tmp_path):
    cfg = tmp_path / "bad.json"
    cfg.write_text(
        """
        {
          "input": {"requirement_file": "r.py", "traces_file": "t.csv"},
          "evaluation": {"trace_check_timeout_sec": 300},
          "heuristics": {
            "two_tier_timeout": {"enabled": true, "low_sec": 60, "high_sec": 600}
          }
        }
        """,
        encoding="utf-8",
    )
    with pytest.raises(ConfigError) as exc:
        load_config(cfg)
    assert "trace_check_timeout_sec" in str(exc.value)


def test_config_rejects_high_below_low(tmp_path):
    cfg = tmp_path / "bad2.json"
    cfg.write_text(
        """
        {
          "input": {"requirement_file": "r.py", "traces_file": "t.csv"},
          "evaluation": {"trace_check_timeout_sec": 3600},
          "heuristics": {
            "two_tier_timeout": {"enabled": true, "low_sec": 600, "high_sec": 60}
          }
        }
        """,
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load_config(cfg)


def test_config_accepts_valid_heuristics(tmp_path):
    cfg = tmp_path / "ok.json"
    cfg.write_text(
        """
        {
          "input": {"requirement_file": "r.py", "traces_file": "t.csv"},
          "evaluation": {"trace_check_timeout_sec": 600},
          "heuristics": {
            "interval_inference": {"enabled": true, "mode": "label", "empirical_validation_k": 5},
            "two_tier_timeout": {"enabled": true, "low_sec": 60, "high_sec": 600}
          }
        }
        """,
        encoding="utf-8",
    )
    loaded = load_config(cfg)
    assert loaded.heuristics.interval_inference.enabled is True
    assert loaded.heuristics.interval_inference.mode == "label"
    assert loaded.heuristics.interval_inference.empirical_validation_k == 5
    assert loaded.heuristics.two_tier_timeout.high_sec == 600
