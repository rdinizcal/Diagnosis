"""Unit tests for the static quantizability gate and period detection.

Covers Feature 1 of Sprint 7: the quantizable/period gate on the shipped AT1
(direct window bound) and AT53 (affine ``i2t(s)+N`` offset) requirements, the
period-mismatch guard, and the polarity promotion of sample-aligned bounds.
"""

from __future__ import annotations

from pathlib import Path

from diagnosis.lang.ast import (
    And,
    ArithOp,
    ForAll,
    Implies,
    IntConst,
    RealConst,
    RelOp,
    Subscript,
    Var,
)
from diagnosis.lang.polarity import Monotonicity, polarity
from diagnosis.lang.quantize import (
    detect_period,
    quantizability,
    quantized_direction,
)
from diagnosis.lang.theodore_parser import load_formula_from_property

EXAMPLES = Path(__file__).resolve().parents[1] / "diagnosis" / "examples"
EFFECT = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "effectiveness"
)


# --------------------------------------------------------------------------
# Shipped examples: AT1 direct window bound (period 10000)
# --------------------------------------------------------------------------


def test_at1_window_bounds_quantizable_period_10000():
    """AT1's window bounds (positions 4, 8) are quantizable with period 10000."""
    formula = load_formula_from_property(EXAMPLES / "AT1_AT001.py")
    assert detect_period(formula) == 10000.0
    for pos in (4, 8):
        info = quantizability(formula, pos)
        assert info.quantizable, info.reason
        assert info.period == 10000.0


def test_at1_signal_threshold_not_quantizable():
    """The speed threshold (position 11) is a signal threshold, not a time bound."""
    info = quantizability(load_formula_from_property(EXAMPLES / "AT1_AT001.py"), 11)
    assert not info.quantizable


def test_at1_window_bounds_get_direction_only_when_quantizing():
    """Window bounds stay UNKNOWN by default but resolve under ``quantize=True``.

    Larger upper bound (pos 8) widens the ForAll interval -> stronger -> the
    SATISFIED verdict is non-increasing (DECREASING); the lower bound (pos 4)
    narrows the interval as it grows -> INCREASING.
    """
    formula = load_formula_from_property(EXAMPLES / "AT1_AT001.py")
    assert polarity(formula, 8) is Monotonicity.UNKNOWN
    assert polarity(formula, 4) is Monotonicity.UNKNOWN
    assert polarity(formula, 8, quantize=True) is Monotonicity.DECREASING
    assert polarity(formula, 4, quantize=True) is Monotonicity.INCREASING
    # The signal threshold direction is unchanged by quantization.
    assert polarity(formula, 11, quantize=True) is Monotonicity.INCREASING


# --------------------------------------------------------------------------
# Shipped examples: AT53 affine i2t(s)+N offset
# --------------------------------------------------------------------------


def test_at53_affine_offset_quantizable_and_directional():
    """AT53's ``t2 <= timestamps[i] + N`` offset is a quantizable affine bound."""
    formula = load_formula_from_property(EFFECT / "AT53" / "AT53_AT119.py")
    assert detect_period(formula) == 10000.0
    # The mutable offset N sits under an affine ``+``; find it as the quantizable
    # position among the config knobs [5, 8, 27].
    quant = [p for p in (5, 8, 27) if quantizability(formula, p).quantizable]
    assert quant, "expected at least one quantizable affine window offset in AT53"
    for p in quant:
        assert quantizability(formula, p).period == 10000.0
        # Quantized promotion yields a definite direction (was UNKNOWN).
        assert polarity(formula, p) is Monotonicity.UNKNOWN
        assert polarity(formula, p, quantize=True) in (
            Monotonicity.INCREASING,
            Monotonicity.DECREASING,
        )


# --------------------------------------------------------------------------
# Constructed structural cases
# --------------------------------------------------------------------------


def _floor_signal(var: str = "t") -> Var:
    """A round-trip-style signal access ``v_speed[ToInt((t-0.0)/10000.0)]``."""
    return Var(f"v_speed[ToInt(RealVal(0)+({var}-0.0)/10000.0)]")


def _window_forall(bound: Formula, op: str = "<=") -> ForAll:
    """``ForAll t. (0 <= t and t <op> bound) -> v_speed[floor(t)] < 120``."""
    interval = And([RelOp("<=", IntConst(0), Var("t")), RelOp(op, Var("t"), bound)])
    cond = RelOp("<", _floor_signal("t"), RealConst(120.0))
    return ForAll(["t"], Implies(interval, cond))


def test_direct_bound_quantizable():
    """A direct ``t <= B`` window bound against a floor-indexed var is quantizable."""
    formula = _window_forall(IntConst(20000000))
    # preorder: 0 ForAll,1 Implies,2 And,3 RelOp,4 IntConst(0),5 Var t,
    #           6 RelOp,7 Var t,8 IntConst(B) -> position 8 is the bound.
    info = quantizability(formula, 8)
    assert info.quantizable and info.period == 10000.0
    assert quantized_direction(formula, 8)[0] is Monotonicity.DECREASING


def test_affine_bound_quantizable():
    """``t <= (5000 + N)`` keeps the constant classifiable through the affine +."""
    formula = _window_forall(ArithOp("+", IntConst(5000), IntConst(30000)))
    # ...6 RelOp,7 Var t,8 ArithOp,9 IntConst(5000),10 IntConst(N=30000).
    assert quantizability(formula, 10).quantizable
    assert quantized_direction(formula, 10)[0] is Monotonicity.DECREASING


def test_multiplicative_offset_not_quantizable():
    """A non-affine (``*``) path breaks affinity -> conservatively not quantizable."""
    formula = _window_forall(ArithOp("*", IntConst(2), IntConst(30000)))
    assert not quantizability(formula, 10).quantizable


def test_dense_time_access_not_quantizable():
    """If the time var also indexes a signal without ToInt, it is dense -> no."""
    interval = And([RelOp("<=", IntConst(0), Var("t")), RelOp("<=", Var("t"), IntConst(9))])
    # v_speed[t] is a dense (non-floor) access of the same variable.
    cond = RelOp("<", Var("v_speed[t]"), RealConst(120.0))
    formula = ForAll(["t"], Implies(interval, cond))
    assert detect_period(formula) is None
    assert not quantizability(formula, 8).quantizable


def test_signal_threshold_not_quantizable():
    """A genuine signal threshold ``v_speed[floor(t)] < N`` is not a time bound."""
    formula = _window_forall(IntConst(20000000))
    # position 11 is the RealConst(120.0) threshold.
    assert not quantizability(formula, 11).quantizable


def test_period_mismatch_disables_quantization():
    """A trace spacing that disagrees with the index divisor disables the gate."""
    formula = _window_forall(IntConst(20000000))
    ok = quantizability(formula, 8, trace_period=10000.0)
    bad = quantizability(formula, 8, trace_period=5000.0)
    assert ok.quantizable
    assert not bad.quantizable
    assert "disagrees" in bad.reason


def test_period_override_must_cross_check_unless_forced():
    """An override that contradicts the divisor is rejected unless ``force``."""
    formula = _window_forall(IntConst(20000000))
    assert not quantizability(formula, 8, period_override=7000.0).quantizable
    forced = quantizability(formula, 8, period_override=7000.0, force=True)
    assert forced.quantizable and forced.period == 7000.0


def test_unreachable_position_not_quantizable():
    formula = _window_forall(IntConst(20000000))
    assert not quantizability(formula, 999).quantizable
