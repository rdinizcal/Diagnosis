"""Unit tests for Sprint 7 Feature 3: adaptive range on sample-class indices.

Drives the pure class-bisection helper against a mocked step function and checks
the ``ceil(log2(#classes))`` termination bound for both monotone directions, plus
the class-index resampling path of the heuristics controller.
"""

from __future__ import annotations

import math
import random

from diagnosis.config import (
    AdaptiveRangeConfig,
    HeuristicsConfig,
    TimeQuantizationConfig,
)
from diagnosis.inference import (
    HeuristicsController,
    numeric_value_map,
    set_numeric_at_position,
)
from diagnosis.lang.ast import (
    And,
    ForAll,
    Implies,
    IntConst,
    RealConst,
    RelOp,
    Var,
)
from diagnosis.quantization import bisect_probe_class, class_midpoint_value

MADEIT_SAT = "True"
MADEIT_VIOLATED = "False"


def _run_bisection(n_classes: int, boundary: int, orient: int) -> tuple[int, int]:
    """Bisect a step function over ``n_classes`` classes; return (found, #solves).

    Step function: for INCREASING (orient +1) classes ``>= boundary`` are SAT and
    the rest UNSAT; for DECREASING (orient -1) it is mirrored.  Seeds the two
    endpoints as the initial bracket, then repeatedly probes the returned midpoint.
    """
    def verdict_sat(cls: int) -> bool:
        return (cls >= boundary) if orient > 0 else (cls <= boundary)

    lo, hi = 0, n_classes - 1
    sat = {c for c in (lo, hi) if verdict_sat(c)}
    unsat = {c for c in (lo, hi) if not verdict_sat(c)}
    solves = 0
    found = None
    for _ in range(n_classes + 5):  # generous safety bound
        probe, bound = bisect_probe_class(sat, unsat, orient)
        if bound is not None:
            found = bound
            break
        if probe is None:
            break
        solves += 1
        (sat if verdict_sat(probe) else unsat).add(probe)
    return found, solves


def test_bisection_increasing_terminates_in_log2():
    found, solves = _run_bisection(300, boundary=173, orient=1)
    # Boundary class is the lowest-SAT class adjacent to the highest-UNSAT class.
    assert found == 173
    assert solves <= math.ceil(math.log2(300))  # <= 9


def test_bisection_decreasing_terminates_in_log2():
    found, solves = _run_bisection(300, boundary=126, orient=-1)
    # DECREASING: SAT is the lower half; the adjacent boundary class is 127.
    assert found == 127
    assert solves <= math.ceil(math.log2(300))


def test_bisect_needs_both_verdicts():
    assert bisect_probe_class(set(), {0, 1}, 1) == (None, None)
    assert bisect_probe_class({5}, set(), 1) == (None, None)


def test_class_midpoint_value():
    assert class_midpoint_value(5, 10000.0) == 55000.0


# --------------------------------------------------------------------------
# Controller integration: class-index resampling + boundary recording
# --------------------------------------------------------------------------


def _floor_signal() -> Var:
    return Var("v_speed[ToInt(RealVal(0)+(t-0.0)/10000.0)]")


def _seed(bound: int = 500000) -> ForAll:
    interval = And([RelOp("<=", IntConst(0), Var("t")), RelOp("<=", Var("t"), IntConst(bound))])
    cond = RelOp("<", _floor_signal(), RealConst(120.0))
    return ForAll(["t"], Implies(interval, cond))


def _controller() -> HeuristicsController:
    cfg = HeuristicsConfig(
        adaptive_range=AdaptiveRangeConfig(enabled=True, exploration_fraction=0.0),
        time_quantization=TimeQuantizationConfig(enabled=True),
    )
    return HeuristicsController(_seed(), cfg, trace_check_timeout_sec=60, trace_period=10000.0)


def test_controller_promotes_and_samples_on_class_indices():
    ctrl = _controller()
    assert ctrl.quant_on
    # The upper window bound (position 8) is now a monotone knob.
    assert 8 in ctrl._M_set
    seed = _seed()
    # Candidate that moves only position 8; class-index resample returns a class
    # midpoint (an odd multiple of period/2).
    cand = set_numeric_at_position(seed, 8, 123456.0)
    out = ctrl.adapt_candidate(cand, {8: {"numeric": [0.0, 500000.0]}}, random.Random(0))
    assert out is not None
    val = numeric_value_map(out)[8]
    assert (val / 5000.0) % 2 == 1  # (cls + 0.5) * 10000 -> odd * 5000


def test_quant_knob_predicate_gates_the_one_class_override():
    """quant_knob is True only for quantizable positions with quantization on.

    The endpoint-init one-class handler uses this to skip halting the whole run on
    a one-class *window* bound (whose boundary is merely outside the sampled range)
    while keeping report_and_stop for ordinary monotone knobs and when off.
    """
    ctrl = _controller()
    assert ctrl.quant_knob(8) is True      # quantizable window bound
    assert ctrl.quant_knob(11) is False    # the signal threshold is not quantizable
    off = HeuristicsController(
        _seed(),
        HeuristicsConfig(adaptive_range=AdaptiveRangeConfig(enabled=True)),
        trace_check_timeout_sec=60,
        trace_period=10000.0,
    )
    assert off.quant_knob(8) is False       # quantization off -> never a quant knob


def test_controller_records_sample_aligned_boundary():
    ctrl = _controller()
    # The upper window bound (position 8) is DECREASING: a wider window (larger B)
    # is a stronger requirement, so lower classes are SAT and higher are VIOLATED.
    assert ctrl._orientation(8) == -1
    allowed = {8: {"numeric": [0.0, 500000.0]}}
    seed = _seed()
    # Pin the boundary between class 12 (SAT) and class 13 (VIOLATED).
    ctrl.adaptive_observations[8] = {
        "sat": [125000.0],    # class 12
        "unsat": [135000.0],  # class 13
        "unknown": [],
    }
    cand = set_numeric_at_position(seed, 8, 130000.0)
    ctrl.adapt_candidate(cand, allowed, random.Random(0))
    # Boundary recorded as the sample timestamp between the adjacent classes.
    assert ctrl.quant.boundary_at["8"] == 13 * 10000.0
