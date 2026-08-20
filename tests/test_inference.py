"""Unit tests for banded interval inference (Feature 2).

Covers the single-mutation AST-diff detector, three-valued region updates, the
monotonicity-violation disable path, the two-tier region-memory decision, and
the controller's guide/label bookkeeping. No solver is invoked.
"""

from __future__ import annotations

import random

import pytest

from diagnosis.config import (
    AdaptiveRangeConfig,
    HeuristicsConfig,
    IntervalInferenceConfig,
    TwoTierTimeoutConfig,
)
from diagnosis.inference import (
    Decision,
    HeuristicsController,
    MonotoneRegion,
    PositionRegion,
    ast_numeric_diffs,
    ast_single_numeric_diff,
    MADEIT_SAT,
    MADEIT_UNDECIDED,
    MADEIT_VIOLATED,
)
from diagnosis.lang.ast import And, ArithOp, IntConst, Not, RelOp, Subscript, Var
from diagnosis.lang.polarity import Monotonicity

INC = Monotonicity.INCREASING
DEC = Monotonicity.DECREASING


def _sig(name="v_speed"):
    return Subscript(base=Var(name), index=Var("i"))


def _threshold(n):
    """Seed-shaped ``signal < n``; numeric constant is position 2, INCREASING."""
    return RelOp("<", _sig(), IntConst(n))


def _heur(inference=True, mode="guide", k=1, two_tier=False, low=60, high=600):
    return HeuristicsConfig(
        interval_inference=IntervalInferenceConfig(
            enabled=inference, mode=mode, empirical_validation_k=k
        ),
        two_tier_timeout=TwoTierTimeoutConfig(enabled=two_tier, low_sec=low, high_sec=high),
    )


def _adaptive_heur(exploration_fraction=0.15, inference=True):
    return HeuristicsConfig(
        interval_inference=IntervalInferenceConfig(
            enabled=inference, mode="guide", empirical_validation_k=1
        ),
        adaptive_range=AdaptiveRangeConfig(
            enabled=True,
            endpoint_init=False,
            exploration_fraction=exploration_fraction,
        ),
    )


# --------------------------------------------------------------------------
# single-mutation detector
# --------------------------------------------------------------------------


def test_single_numeric_diff_detected():
    seed = _threshold(120)
    cand = _threshold(140)
    assert ast_single_numeric_diff(seed, cand) == (2, 140.0)


def test_identical_formula_no_diff():
    seed = _threshold(120)
    assert ast_single_numeric_diff(seed, _threshold(120)) is None


def test_two_position_mutant_not_inferred():
    """A crossover offspring changing two numbers must NOT be treated as single."""
    seed = And([_threshold(120), RelOp(">", _sig("e"), IntConst(3000))])
    cand = And([_threshold(140), RelOp(">", _sig("e"), IntConst(3500))])
    assert ast_single_numeric_diff(seed, cand) is None


def test_ast_numeric_diffs_returns_all_changes():
    """The k-D detector returns every changed numeric position."""
    seed = And([_threshold(120), RelOp(">", _sig("e"), IntConst(3000))])
    cand = And([_threshold(140), RelOp(">", _sig("e"), IntConst(3500))])
    # positions: 0 And,1 RelOp,2 Subscript,3 IntConst(120@pos3),
    #            4 RelOp,5 Subscript,6 IntConst(3000@pos6)
    diffs = ast_numeric_diffs(seed, cand)
    assert diffs == [(3, 140.0), (6, 3500.0)]
    # structural change still -> None
    assert ast_numeric_diffs(seed, Not(seed)) is None
    # identical -> empty list
    assert ast_numeric_diffs(seed, seed) == []


def test_operator_change_not_numeric_diff():
    seed = _threshold(120)
    cand = RelOp(">", _sig(), IntConst(120))
    assert ast_single_numeric_diff(seed, cand) is None


def test_structural_change_not_numeric_diff():
    seed = _threshold(120)
    cand = Not(_threshold(120))
    assert ast_single_numeric_diff(seed, cand) is None


# --------------------------------------------------------------------------
# region logic (three-valued)
# --------------------------------------------------------------------------


def test_increasing_region_infers_halflines():
    r = PositionRegion(Monotonicity.INCREASING)
    r.update(100.0, MADEIT_VIOLATED)
    r.update(140.0, MADEIT_SAT)
    # k satisfied (2 consistent solves); above sat_min -> SAT, below unsat_max -> VIOLATED
    assert r.decide(150.0, k=2) is Decision.INFER_SAT
    assert r.decide(90.0, k=2) is Decision.INFER_VIOLATED
    # inside the frontier -> must solve
    assert r.decide(120.0, k=2) is Decision.SOLVE_NORMAL


def test_decreasing_region_mirrors():
    r = PositionRegion(Monotonicity.DECREASING)
    r.update(100.0, MADEIT_SAT)     # satisfied at low values
    r.update(140.0, MADEIT_VIOLATED)
    assert r.decide(90.0, k=2) is Decision.INFER_SAT
    assert r.decide(150.0, k=2) is Decision.INFER_VIOLATED
    assert r.decide(120.0, k=2) is Decision.SOLVE_NORMAL


def test_k_gate_blocks_inference_until_met():
    r = PositionRegion(Monotonicity.INCREASING)
    r.update(140.0, MADEIT_SAT)  # 1 consistent solve
    assert r.decide(150.0, k=3) is Decision.SOLVE_NORMAL  # k not yet met
    r.update(141.0, MADEIT_SAT)
    r.update(142.0, MADEIT_SAT)
    assert r.decide(150.0, k=3) is Decision.INFER_SAT


def test_undecided_band_bracketed_and_near():
    r = PositionRegion(Monotonicity.INCREASING)
    r.update(118.0, MADEIT_UNDECIDED)
    r.update(122.0, MADEIT_UNDECIDED)
    # bracketed by two undecided observations -> infer UNDECIDED (no solve)
    assert r.decide(120.0, k=1) is Decision.INFER_UNDECIDED
    r2 = PositionRegion(Monotonicity.INCREASING)
    r2.update(120.0, MADEIT_UNDECIDED)  # single band point
    # near but not bracketed -> solve low tier only (region memory)
    assert r2.decide(120.0, k=1) is Decision.SOLVE_LOW_ONLY


def test_monotonicity_violation_disables_region():
    r = PositionRegion(Monotonicity.INCREASING)
    r.update(140.0, MADEIT_SAT)
    witness = r.update(150.0, MADEIT_VIOLATED)  # VIOLATED above a SATISFIED value
    assert witness is not None
    assert r.disabled is True
    assert r.sat_values == [] and r.unsat_values == []
    # A disabled region infers nothing.
    assert r.decide(200.0, k=0) is Decision.SOLVE_NORMAL


def test_frontier_midpoint():
    r = PositionRegion(Monotonicity.INCREASING)
    r.update(100.0, MADEIT_VIOLATED)
    r.update(140.0, MADEIT_SAT)
    assert r.frontier_midpoint(min_gap=1e-6) == pytest.approx(120.0)
    # too narrow -> None
    r2 = PositionRegion(Monotonicity.INCREASING)
    r2.update(120.0, MADEIT_VIOLATED)
    r2.update(120.0 + 1e-9, MADEIT_SAT)
    assert r2.frontier_midpoint(min_gap=1e-6) is None


# --------------------------------------------------------------------------
# controller
# --------------------------------------------------------------------------


def test_controller_infers_after_real_solve_guide_excludes_from_arff():
    ctrl = HeuristicsController(_threshold(120), _heur(k=1), trace_check_timeout_sec=600)
    # First candidate (140): frontier -> solve
    plan1 = ctrl.plan(_threshold(140))
    assert plan1.inferred is None and plan1.position == 2 and plan1.value == 140.0
    ctrl.record_solve(plan1, [MADEIT_SAT], MADEIT_SAT)
    # Second candidate (150): inferred SAT, excluded from ARFF in guide mode
    plan2 = ctrl.plan(_threshold(150))
    assert plan2.inferred == MADEIT_SAT
    assert plan2.inferred_flag is True
    assert plan2.include_in_arff is False
    assert ctrl.counters["inferred_satisfied"] == 1


def test_controller_label_mode_includes_inferred_rows():
    ctrl = HeuristicsController(_threshold(120), _heur(mode="label", k=1), trace_check_timeout_sec=600)
    plan1 = ctrl.plan(_threshold(140))
    ctrl.record_solve(plan1, [MADEIT_SAT], MADEIT_SAT)
    plan2 = ctrl.plan(_threshold(150))
    assert plan2.inferred == MADEIT_SAT
    assert plan2.include_in_arff is True  # label keeps inferred rows


def test_controller_records_monotonicity_violation():
    ctrl = HeuristicsController(_threshold(120), _heur(k=0), trace_check_timeout_sec=600)
    p1 = ctrl.plan(_threshold(140))
    ctrl.record_solve(p1, [MADEIT_SAT], MADEIT_SAT)
    # Inject an inconsistent real solve by forcing a solve plan on a lower value.
    # With k=0 the controller would infer, so exercise the region guard directly
    # through record_solve on a fabricated plan.
    from diagnosis.inference import Plan
    bad = Plan(vector=[150.0], positions=[2], timeouts=[600])
    ctrl.record_solve(bad, [MADEIT_VIOLATED], MADEIT_VIOLATED)
    assert ctrl.counters["monotonicity_violations"] == 1
    assert ctrl.region.disabled is True


def test_two_tier_timeouts_and_region_memory():
    ctrl = HeuristicsController(
        _threshold(120), _heur(inference=True, k=1, two_tier=True, low=60, high=600),
        trace_check_timeout_sec=3600,
    )
    # Non-band candidate -> two tiers.
    plan = ctrl.plan(_threshold(133))
    assert plan.timeouts == [60, 600]
    # Establish an UNDECIDED band point at 133, then a candidate near it -> low only.
    ctrl.record_solve(plan, [MADEIT_UNDECIDED, MADEIT_UNDECIDED], MADEIT_UNDECIDED)
    plan2 = ctrl.plan(_threshold(133))
    assert plan2.timeouts == [60]  # region memory skips the high tier
    assert ctrl.counters["region_memory_skips"] == 1


def test_two_tier_disabled_uses_trace_timeout():
    ctrl = HeuristicsController(_threshold(120), _heur(inference=False, two_tier=False), trace_check_timeout_sec=1234)
    plan = ctrl.plan(_threshold(140))
    assert plan.timeouts == [1234]


# --------------------------------------------------------------------------
# k-D Pareto-domination region (multi-knob)
# --------------------------------------------------------------------------


def test_monotone_region_1d_matches_positionregion():
    """MonotoneRegion with one INCREASING knob behaves like the 1-D region."""
    r = MonotoneRegion([0], [INC])
    r.update([100.0], MADEIT_VIOLATED)
    r.update([140.0], MADEIT_SAT)
    assert r.decide([150.0], k=2) is Decision.INFER_SAT
    assert r.decide([90.0], k=2) is Decision.INFER_VIOLATED
    assert r.decide([120.0], k=2) is Decision.SOLVE_NORMAL


def test_kd_domination_infers_by_pareto_order():
    """Two INCREASING knobs: dominate a SAT point -> SAT; dominated by VIOLATED -> VIOLATED."""
    r = MonotoneRegion([3, 6], [INC, INC])
    r.update([120.0, 3000.0], MADEIT_SAT)
    r.update([100.0, 2000.0], MADEIT_VIOLATED)
    # dominates the SAT corner in both coords -> SAT
    assert r.decide([130.0, 3100.0], k=2) is Decision.INFER_SAT
    # dominated by the VIOLATED corner in both coords -> VIOLATED
    assert r.decide([90.0, 1500.0], k=2) is Decision.INFER_VIOLATED
    # incomparable (above on one axis, below on the other) -> must solve
    assert r.decide([130.0, 1500.0], k=2) is Decision.SOLVE_NORMAL
    assert r.decide([90.0, 3100.0], k=2) is Decision.SOLVE_NORMAL


def test_kd_mixed_direction_orientation():
    """One INCREASING, one DECREASING knob orients each axis independently."""
    # SAT at (high x0, low x1); VIOLATED at (low x0, high x1)
    r = MonotoneRegion([0, 1], [INC, DEC])
    r.update([120.0, 10.0], MADEIT_SAT)
    r.update([100.0, 50.0], MADEIT_VIOLATED)
    # higher x0 and lower x1 -> more satisfied -> SAT
    assert r.decide([130.0, 5.0], k=2) is Decision.INFER_SAT
    # lower x0 and higher x1 -> more violated -> VIOLATED
    assert r.decide([90.0, 60.0], k=2) is Decision.INFER_VIOLATED


def test_kd_runtime_guard_disables_on_dominating_violation():
    """A VIOLATED point that dominates a SAT point is a monotonicity violation."""
    r = MonotoneRegion([0, 1], [INC, INC])
    r.update([120.0, 3000.0], MADEIT_SAT)
    # (130, 3100) dominates the SAT point but comes back VIOLATED -> impossible
    witness = r.update([130.0, 3100.0], MADEIT_VIOLATED)
    assert witness is not None
    assert r.disabled is True
    assert r.decide([200.0, 9000.0], k=0) is Decision.SOLVE_NORMAL


def _two_knob_seed():
    """And(a < N1, b > N2): positions 3 (a<N1, INCREASING), 6 (b>N2, DECREASING)."""
    return And([
        RelOp("<", _sig("a"), IntConst(120)),
        RelOp(">", _sig("b"), IntConst(3000)),
    ])


def test_controller_infers_multiple_knobs():
    """A 2-knob seed: joint region infers a candidate that moved both knobs."""
    ctrl = HeuristicsController(_two_knob_seed(), _heur(k=1), trace_check_timeout_sec=600)
    assert ctrl._M == [3, 6]  # both knobs monotone
    # solve two bracketing corners (both knobs changed at once)
    c_sat = And([RelOp("<", _sig("a"), IntConst(140)), RelOp(">", _sig("b"), IntConst(2000))])
    c_vio = And([RelOp("<", _sig("a"), IntConst(100)), RelOp(">", _sig("b"), IntConst(4000))])
    p1 = ctrl.plan(c_sat)
    assert p1.inferred is None and p1.vector == [140.0, 2000.0]
    ctrl.record_solve(p1, [MADEIT_SAT], MADEIT_SAT)
    p2 = ctrl.plan(c_vio)
    ctrl.record_solve(p2, [MADEIT_VIOLATED], MADEIT_VIOLATED)
    # a candidate dominating the SAT corner (a even larger, N2 even smaller) -> inferred SAT
    c_infer = And([RelOp("<", _sig("a"), IntConst(150)), RelOp(">", _sig("b"), IntConst(1500))])
    p3 = ctrl.plan(c_infer)
    assert p3.inferred == MADEIT_SAT
    assert ctrl.counters["inferred_satisfied"] == 1


def test_controller_skips_inference_when_unknown_knob_moves():
    """If a candidate also moves a non-monotone numeric knob, it is solved."""
    # seed: And(a<120, gear==4); position 6 (gear==4) is UNKNOWN polarity (equality)
    seed = And([RelOp("<", _sig("a"), IntConst(120)), RelOp("==", _sig("g"), IntConst(4))])
    ctrl = HeuristicsController(seed, _heur(k=1), trace_check_timeout_sec=600)
    assert ctrl._M == [3]  # only the a<N knob is monotone
    # moving both the monotone knob and the equality knob -> not inferable
    cand = And([RelOp("<", _sig("a"), IntConst(140)), RelOp("==", _sig("g"), IntConst(5))])
    p = ctrl.plan(cand)
    assert p.inferred is None and p.vector is None  # went to solver


# --------------------------------------------------------------------------
# adaptive mutation range
# --------------------------------------------------------------------------


def test_adaptive_range_samples_from_single_position_bracket():
    ctrl = HeuristicsController(
        _threshold(120), _adaptive_heur(exploration_fraction=0.0),
        trace_check_timeout_sec=600,
    )
    allowed = {2: {"numeric": [0.0, 200.0]}}
    low = ctrl.plan(_threshold(100))
    ctrl.record_solve(low, [MADEIT_VIOLATED], MADEIT_VIOLATED)
    high = ctrl.plan(_threshold(140))
    ctrl.record_solve(high, [MADEIT_SAT], MADEIT_SAT)

    adapted = ctrl.adapt_candidate(_threshold(180), allowed, random.Random(7))
    assert adapted is not None
    assert ast_single_numeric_diff(_threshold(120), adapted)[0] == 2
    value = ast_single_numeric_diff(_threshold(120), adapted)[1]
    assert 100.0 <= value <= 140.0
    assert ctrl.report()["adaptive_draws_bracket"] == 1


def test_adaptive_range_exploration_draw_uses_full_range():
    ctrl = HeuristicsController(
        _threshold(120), _adaptive_heur(exploration_fraction=1.0),
        trace_check_timeout_sec=600,
    )
    allowed = {2: {"numeric": [0.0, 200.0]}}
    ctrl.record_solve(ctrl.plan(_threshold(100)), [MADEIT_VIOLATED], MADEIT_VIOLATED)
    ctrl.record_solve(ctrl.plan(_threshold(140)), [MADEIT_SAT], MADEIT_SAT)

    adapted = ctrl.adapt_candidate(_threshold(180), allowed, random.Random(2))
    assert adapted is not None
    assert ctrl.report()["adaptive_draws_exploration"] == 1


def test_adaptive_range_records_unknown_band_and_falls_back_to_full_range():
    ctrl = HeuristicsController(
        _threshold(120), _adaptive_heur(exploration_fraction=0.0),
        trace_check_timeout_sec=600,
    )
    allowed = {2: {"numeric": [0.0, 200.0]}}
    ctrl.record_solve(ctrl.plan(_threshold(118)), [MADEIT_UNDECIDED], MADEIT_UNDECIDED)
    ctrl.record_solve(ctrl.plan(_threshold(122)), [MADEIT_UNDECIDED], MADEIT_UNDECIDED)

    adapted = ctrl.adapt_candidate(_threshold(180), allowed, random.Random(4))
    assert adapted is not None
    report = ctrl.report()
    assert report["adaptive_unknown_bands"][0]["interval"] == [118.0, 122.0]
    assert report["adaptive_draws_full_fallback"] == 1


def test_adaptive_range_ignores_multi_position_solve_for_brackets():
    ctrl = HeuristicsController(
        _two_knob_seed(), _adaptive_heur(exploration_fraction=0.0),
        trace_check_timeout_sec=600,
    )
    from diagnosis.inference import Plan

    multi = Plan(vector=[140.0, 2000.0], positions=[3, 6], timeouts=[600])
    ctrl.record_solve(multi, [MADEIT_SAT], MADEIT_SAT)
    assert ctrl.report()["adaptive_observations"] == {}
