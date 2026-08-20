"""Unit tests for Sprint 7 Feature 2: canonical cache keys, the runtime
validator, and the vacuity guard (the :class:`diagnosis.quantization.Quantizer`).
"""

from __future__ import annotations

from diagnosis.cache import expression_key
from diagnosis.config import TimeQuantizationConfig
from diagnosis.inference import set_numeric_at_position
from diagnosis.lang.ast import (
    And,
    ForAll,
    Implies,
    IntConst,
    RealConst,
    RelOp,
    Var,
)
from diagnosis.lang.python_printer import formula_to_python_expr
from diagnosis.quantization import Quantizer, trace_period_from_property


def _floor_signal() -> Var:
    return Var("v_speed[ToInt(RealVal(0)+(t-0.0)/10000.0)]")


def _seed(bound: int = 50000) -> ForAll:
    """``ForAll t. (0 <= t and t <= bound) -> v_speed[floor(t)] < 120``.

    Window bounds at positions 4 (lower) and 8 (upper) are quantizable with
    period 10000; the threshold (position 11) is a raw signal threshold.
    """
    interval = And([RelOp("<=", IntConst(0), Var("t")), RelOp("<=", Var("t"), IntConst(bound))])
    cond = RelOp("<", _floor_signal(), RealConst(120.0))
    return ForAll(["t"], Implies(interval, cond))


def _cfg(**kw) -> TimeQuantizationConfig:
    return TimeQuantizationConfig(enabled=True, **kw)


def _cand(seed, position, value):
    return set_numeric_at_position(seed, position, value)


def _raw(ast) -> str:
    return formula_to_python_expr(ast)


# --------------------------------------------------------------------------
# Canonicalization
# --------------------------------------------------------------------------


def test_same_class_values_share_a_key():
    """51237.4 / 54003.9 / 57300.0 collapse to one class (5); 63412.7 is class 6."""
    seed = _seed()
    q = Quantizer(seed, _cfg())
    assert q.enabled
    keys = []
    for v in (51237.4, 54003.9, 57300.0, 63412.7):
        cand = _cand(seed, 8, v)
        keys.append(expression_key(q.canonical_expr(cand, _raw(cand))))
    assert keys[0] == keys[1] == keys[2]
    assert keys[3] != keys[0]


def test_mixed_candidate_keeps_threshold_raw():
    """A raw (non-quantizable) threshold participates in the key un-canonicalized."""
    seed = _seed()
    q = Quantizer(seed, _cfg())
    # Mutate both the quantizable bound (pos 8) and the raw threshold (pos 11).
    cand = _cand(_cand(seed, 8, 54003.9), 11, 99.0)
    canon = q.canonical_expr(cand, _raw(cand))
    assert "99.0" in canon        # threshold stays raw
    assert "54003" not in canon   # the time bound is replaced by its class index


def test_disabled_when_flag_off_is_identity():
    """With quantization off the canonical expression is exactly the raw one."""
    seed = _seed()
    q = Quantizer(seed, TimeQuantizationConfig(enabled=False))
    assert not q.enabled
    cand = _cand(seed, 8, 54003.9)
    assert q.canonical_expr(cand, _raw(cand)) == _raw(cand)


# --------------------------------------------------------------------------
# Runtime validator
# --------------------------------------------------------------------------


def test_validator_disables_position_and_purges_on_mismatch():
    seed = _seed()
    q = Quantizer(seed, _cfg(validate_every_n_hits=1))
    cand_a = _cand(seed, 8, 51000.0)   # class 5
    cand_b = _cand(seed, 8, 55000.0)   # same class 5, different raw
    canon = q.canonical_expr(cand_a, _raw(cand_a))
    assert canon == q.canonical_expr(cand_b, _raw(cand_b))

    # First raw solved (miss records it); second raw is a canonical hit -> due.
    assert q.on_lookup(canon, _raw(cand_a), hit=False) is False
    assert q.on_lookup(canon, _raw(cand_b), hit=True) is True

    purge = q.record_validation(cand_b, canon, cached_verdict="True", real_verdict="False")
    assert q.violations == 1
    assert 8 in q.disabled
    assert purge  # class entries to evict
    assert q.witnesses and q.witnesses[0]["cached_verdict"] == "True"
    # After disabling, the position is no longer canonicalized.
    assert q.canonical_expr(cand_b, _raw(cand_b)) == _raw(cand_b)


def test_validator_no_action_when_verdicts_agree():
    seed = _seed()
    q = Quantizer(seed, _cfg(validate_every_n_hits=1))
    cand = _cand(seed, 8, 55000.0)
    canon = q.canonical_expr(cand, _raw(cand))
    assert q.record_validation(cand, canon, "True", "True") == []
    assert q.violations == 0
    assert not q.disabled


def test_exact_vs_canonical_hit_counting():
    seed = _seed()
    q = Quantizer(seed, _cfg(validate_every_n_hits=0))
    cand_a = _cand(seed, 8, 51000.0)
    cand_b = _cand(seed, 8, 55000.0)
    canon = q.canonical_expr(cand_a, _raw(cand_a))
    q.on_lookup(canon, _raw(cand_a), hit=False)   # first raw recorded
    q.on_lookup(canon, _raw(cand_a), hit=True)    # same raw -> exact hit
    q.on_lookup(canon, _raw(cand_b), hit=True)    # new raw, same class -> canonical
    assert q.exact_hits == 1
    assert q.canonical_hits == 1


# --------------------------------------------------------------------------
# Vacuity guard
# --------------------------------------------------------------------------


def test_vacuous_window_flagged_and_counted():
    seed = _seed()
    q = Quantizer(seed, _cfg())
    assert q.window_pairs  # lower pos 4, upper pos 8
    # lower (60000) >= upper (50000) -> empty window.
    vac = _cand(_cand(seed, 4, 60000), 8, 50000)
    assert q.vacuous_flag(vac) is True
    assert q.vacuous_candidates == 1
    # A normal ordered window is not vacuous.
    ok = _cand(_cand(seed, 4, 0), 8, 50000)
    assert q.vacuous_flag(ok) is False
    assert q.vacuous_candidates == 1


# --------------------------------------------------------------------------
# Trace period detection
# --------------------------------------------------------------------------


def test_trace_period_from_property(tmp_path):
    p = tmp_path / "prop.py"
    p.write_text(
        "z3solver.add(timestamps[ 0]==0)\n"
        "z3solver.add(timestamps[ 1]==10000)\n"
        "z3solver.add(timestamps[ 2]==20000)\n",
        encoding="utf-8",
    )
    assert trace_period_from_property(str(p)) == 10000.0
