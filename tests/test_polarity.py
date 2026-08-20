"""Unit tests for static polarity (monotonicity) analysis.

Covers the shipped example requirements and constructed edge cases:
INCREASING thresholds, sign flips through an Implies antecedent, and the
conservative UNKNOWN paths (equality, arithmetic, temporal bounds, mixed
polarity).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from diagnosis.lang.ast import (
    And,
    ArithOp,
    ForAll,
    Implies,
    IntConst,
    Not,
    Or,
    RealConst,
    RelOp,
    Subscript,
    Var,
)
from diagnosis.lang.polarity import (
    Monotonicity,
    numeric_positions,
    parameter_polarity,
    polarity,
    polarity_with_reason,
)
from diagnosis.lang.theodore_parser import load_formula_from_property

EXAMPLES = Path(__file__).resolve().parents[1] / "diagnosis" / "examples"
EFFECT = (
    Path(__file__).resolve().parents[1]
    / "replication"
    / "evaluation_inputs"
    / "effectiveness"
)


def _signal(name: str = "v_speed") -> Subscript:
    """A stand-in for a signal access such as ``v_speed[ToInt(...)]``."""
    return Subscript(base=Var(name), index=Var("i"))


# --------------------------------------------------------------------------
# Shipped examples
# --------------------------------------------------------------------------


def test_at1_speed_threshold_is_increasing():
    """AT1: ``v_speed[...] < 120.0`` at position 11 is INCREASING."""
    formula = load_formula_from_property(EXAMPLES / "AT1_AT001.py")
    assert polarity(formula, 11) is Monotonicity.INCREASING
    # The time-window bounds are conservatively UNKNOWN.
    assert polarity(formula, 4) is Monotonicity.UNKNOWN
    assert polarity(formula, 8) is Monotonicity.UNKNOWN


def test_at2_speed_threshold_is_increasing():
    """AT2 has the same shape as AT1; the threshold is INCREASING."""
    formula = load_formula_from_property(EXAMPLES / "AT2_AT015.py")
    assert polarity(formula, 11) is Monotonicity.INCREASING


def test_at6b_sign_flows_through_implies_antecedent():
    """AT6B is ``Implies(A, B)`` (outer Not stripped by the parser).

    The consequent threshold (position 24, ``v_speed < 50``) is INCREASING; the
    antecedent threshold (position 12, ``e_speed < 3000``) is DECREASING because
    the Implies antecedent flips polarity.
    """
    formula = load_formula_from_property(EFFECT / "AT6B" / "AT6B_AT272.py")
    assert isinstance(formula, Implies)
    assert polarity(formula, 24) is Monotonicity.INCREASING
    assert polarity(formula, 12) is Monotonicity.DECREASING
    # Temporal bounds remain UNKNOWN on both sides.
    for pos in (5, 9, 17, 21):
        assert polarity(formula, pos) is Monotonicity.UNKNOWN


def test_cc5_thresholds_classified_bounds_unknown():
    """CC5 separation thresholds are monotone; nested temporal bounds are not."""
    formula = load_formula_from_property(EFFECT / "CC5" / "CC5_CC031.py")
    assert polarity(formula, 36) is Monotonicity.INCREASING
    assert polarity(formula, 54) is Monotonicity.DECREASING
    # Constants nested under arithmetic (window offsets) are UNKNOWN.
    for pos in (19, 31, 43, 49):
        assert polarity(formula, pos) is Monotonicity.UNKNOWN


def test_numeric_positions_match_layout():
    """``numeric_positions`` finds exactly the AT1 numeric constants."""
    formula = load_formula_from_property(EXAMPLES / "AT1_AT001.py")
    assert numeric_positions(formula) == [4, 8, 11]


def test_polarity_stable_across_internal_roundtrip():
    """The GA's seed AST comes from an internal encode/decode round-trip that
    renders signal subscripts as string Vars. Polarity must be identical to the
    direct parse (positions and directions), or the controller would disable
    inference where the CLI reports it as monotone.
    """
    from diagnosis.lang.internal_encoder import formula_to_internal_obj
    from diagnosis.lang.internal_parser import parse_internal_obj

    direct = load_formula_from_property(EXAMPLES / "AT1_AT001.py")
    roundtrip = parse_internal_obj(formula_to_internal_obj(direct))
    assert numeric_positions(direct) == numeric_positions(roundtrip)
    for p in numeric_positions(direct):
        assert polarity(direct, p) is polarity(roundtrip, p)
    assert polarity(roundtrip, 11) is Monotonicity.INCREASING


# --------------------------------------------------------------------------
# Constructed structural cases
# --------------------------------------------------------------------------


def test_greater_than_is_decreasing():
    """``signal > N`` with a preserved sign flag is DECREASING."""
    # positions: 0 RelOp, 1 Subscript, 2 IntConst
    formula = RelOp(">", _signal(), IntConst(10))
    assert polarity(formula, 2) is Monotonicity.DECREASING


def test_constant_on_left_inverts():
    """``N < signal`` (constant on the left) inverts to DECREASING."""
    # positions: 0 RelOp, 1 IntConst, 2 Subscript
    formula = RelOp("<", IntConst(10), _signal())
    assert polarity(formula, 1) is Monotonicity.DECREASING


def test_single_not_flips_direction():
    """A single Not flips an INCREASING threshold to DECREASING."""
    # positions: 0 Not, 1 RelOp, 2 Subscript, 3 IntConst
    formula = Not(RelOp("<", _signal(), IntConst(10)))
    assert polarity(formula, 3) is Monotonicity.DECREASING


def test_equality_is_unknown():
    """Equality/inequality comparisons are non-monotone -> UNKNOWN."""
    formula = RelOp("==", _signal(), IntConst(10))
    assert polarity(formula, 2) is Monotonicity.UNKNOWN


def test_constant_under_arithmetic_is_unknown():
    """A constant nested inside arithmetic is not a classifiable threshold."""
    # signal < (5 + N): N is under an ArithOp, not a direct RelOp operand.
    # positions: 0 RelOp, 1 Subscript, 2 ArithOp, 3 IntConst(5), 4 IntConst(N)
    formula = RelOp("<", _signal(), ArithOp("+", IntConst(5), IntConst(99)))
    assert polarity(formula, 4) is Monotonicity.UNKNOWN


def test_no_signal_term_is_unknown():
    """A comparison with no signal term is treated as a temporal bound."""
    formula = RelOp("<=", IntConst(0), Var("Real('t')"))
    assert polarity(formula, 1) is Monotonicity.UNKNOWN


def test_unreachable_position_is_unknown():
    """An out-of-range position is UNKNOWN, not an error."""
    formula = RelOp("<", _signal(), IntConst(10))
    d, reason = polarity_with_reason(formula, 999)
    assert d is Monotonicity.UNKNOWN
    assert "reachable" in reason


def test_mixed_polarity_parameter_is_unknown():
    """A parameter reachable with both signs is not monotone overall.

    The same threshold value 10 appears once positively (``s < 10``) and once
    under a Not (``Not(s < 10)``); combining the two positions yields UNKNOWN.
    """
    # And([ s<10 , Not(s<10) ])
    # preorder: 0 And, 1 RelOp, 2 Subscript, 3 IntConst(pos A),
    #           4 Not, 5 RelOp, 6 Subscript, 7 IntConst(pos B)
    formula = And([
        RelOp("<", _signal(), IntConst(10)),
        Not(RelOp("<", _signal(), IntConst(10))),
    ])
    assert polarity(formula, 3) is Monotonicity.INCREASING
    assert polarity(formula, 7) is Monotonicity.DECREASING
    assert parameter_polarity(formula, [3, 7]) is Monotonicity.UNKNOWN
    # A single consistent occurrence still resolves.
    assert parameter_polarity(formula, [3]) is Monotonicity.INCREASING


def test_or_and_preserve_polarity():
    """And/Or preserve the polarity flag for their arguments."""
    formula = Or([
        RelOp("<", _signal("a"), IntConst(1)),
        And([RelOp(">", _signal("b"), RealConst(2.0))]),
    ])
    # position of IntConst(1): 0 Or,1 RelOp,2 Subscript,3 IntConst -> INCREASING
    assert polarity(formula, 3) is Monotonicity.INCREASING
    # position of RealConst(2.0): 4 And,5 RelOp,6 Subscript,7 RealConst -> DECREASING
    assert polarity(formula, 7) is Monotonicity.DECREASING
