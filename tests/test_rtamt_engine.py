"""Unit tests for the lightweight RTAMT trace-checking engine (engine='rtamt').

These exercise the pieces that are specific to the RTAMT boundary: per-subject
template rendering (incl. the single time-unit conversion), the robustness-sign
to verdict mapping (incl. rho==0), the constant-period assertion, signal-to-column
mapping, and hand-computed CCX verdicts derived directly from the trace.
"""

import copy
import textwrap

import pytest

pytest.importorskip("rtamt")

from diagnosis.lang.theodore_parser import load_formula_from_property
from diagnosis.lang.internal_encoder import _collect_positions
from diagnosis.engines.rtamt_engine import (
    RtamtEngine,
    RtamtEngineError,
    RtamtUnavailable,
    Subject,
    REGISTRY,
    subject_from_path,
    _load_trace,
    V_SATISFIED,
    V_VIOLATED,
    V_UNDECIDED,
)

AT1_PATH = "replication/evaluation_inputs/effectiveness/AT1/AT1_AT001.py"
CCX_PATH = "replication/evaluation_inputs/effectiveness/CCX/CCX_CC041.py"


def _set(formula, pos_value):
    """Return a copy of ``formula`` with numeric literals at given positions set."""
    a = copy.deepcopy(formula)
    positions = _collect_positions(a)
    for pos, value in pos_value.items():
        object.__setattr__(positions[int(pos)].node, "value", value)
    return a


@pytest.fixture(scope="module")
def at1_formula():
    return load_formula_from_property(AT1_PATH)


@pytest.fixture(scope="module")
def at1_engine():
    return RtamtEngine(AT1_PATH, subject="AT1")


@pytest.fixture(scope="module")
def ccx_formula():
    return load_formula_from_property(CCX_PATH)


@pytest.fixture(scope="module")
def ccx_engine():
    return RtamtEngine(CCX_PATH, subject="CCX")


# --- trace loading / signal mapping -------------------------------------------

def test_load_trace_constant_period_and_signals():
    signals, step_us, period_s, n = _load_trace(AT1_PATH, ("v_speed",))
    assert step_us == 10000.0
    assert period_s == pytest.approx(0.01)
    assert n > 2000
    # signal-to-column mapping: v_speed values come from the property verbatim.
    assert signals["v_speed"][0] == pytest.approx(0.0)
    assert signals["v_speed"][1] == pytest.approx(0.057361185)


def test_load_trace_maps_all_ccx_signals():
    signals, _, _, _ = _load_trace(CCX_PATH, ("y1", "y2", "y3", "y4", "y5"))
    # First sample of the CC trace: y = [0, 10, 20, 30, 40].
    assert [signals[f"y{i}"][0] for i in range(1, 6)] == [0.0, 10.0, 20.0, 30.0, 40.0]


def test_non_constant_period_raises(tmp_path):
    prop = tmp_path / "bad.py"
    prop.write_text(
        textwrap.dedent(
            """
            z3solver.add(timestamps[0]==0)
            z3solver.add(v_speed[0]==1.0)
            z3solver.add(timestamps[1]==10000)
            z3solver.add(v_speed[1]==1.0)
            z3solver.add(timestamps[2]==25000)
            z3solver.add(v_speed[2]==1.0)
            """
        )
    )
    with pytest.raises(RtamtEngineError, match="non-constant sampling period"):
        _load_trace(prop, ("v_speed",))


# --- template rendering (incl. the single time-unit conversion) ---------------

def test_render_at1_three_positions(at1_engine, at1_formula):
    # lo=0 s, hi=20 s (=20_000_000 us), thr=120 -> samples [0:2000].
    cand = _set(at1_formula, {4: 0, 8: 20_000_000, 11: 120.0})
    assert at1_engine.render(cand) == "out = always[0:2000] (v_speed < 120)"


def test_render_at1_unit_conversion_floors_like_toint(at1_engine, at1_formula):
    # A non-multiple hi floors to the same sample index Z3's ToInt would pick.
    cand = _set(at1_formula, {4: 5_000_000, 8: 20_019_999, 11: 133.5})
    assert at1_engine.render(cand) == "out = always[500:2001] (v_speed < 133.5)"


def test_render_at1_empty_window_is_vacuous(at1_engine, at1_formula):
    # lo > hi -> empty universally-quantified window -> vacuously satisfied.
    cand = _set(at1_formula, {4: 9_000_000, 8: 1_000_000, 11: 120.0})
    assert at1_engine.render(cand) is None
    verdict, rho = at1_engine.check(cand)
    assert verdict == V_SATISFIED
    assert rho == float("inf")


def test_render_ccx_single_position(ccx_engine, ccx_formula):
    # Only the y5-y4 threshold (position 16) is mutable; the rest is verbatim.
    cand = _set(ccx_formula, {16: 6.0})
    rendered = ccx_engine.render(cand)
    assert rendered == (
        "out = (always[0:5000] (y5 - y4 > 6)) "
        "and (always[0:5000] (y4 - y3 > 7.5)) "
        "and (always[0:5000] (y3 - y2 > 7.5)) "
        "and (always[0:5000] (y2 - y1 > 7.5))"
    )


# --- robustness-sign to verdict mapping (incl. rho == 0) ----------------------

@pytest.mark.parametrize(
    "rho,expected",
    [(2.5, V_SATISFIED), (-2.5, V_VIOLATED), (0.0, V_UNDECIDED)],
)
def test_verdict_mapping_from_rho(at1_engine, at1_formula, monkeypatch, rho, expected):
    monkeypatch.setattr(at1_engine, "_robustness", lambda spec: rho)
    before = at1_engine.rho_zero
    cand = _set(at1_formula, {4: 0, 8: 20_000_000, 11: 120.0})
    verdict, out = at1_engine.check(cand)
    assert verdict == expected
    assert out == rho
    assert at1_engine.rho_zero == before + (1 if rho == 0.0 else 0)


# --- CCX hand-computed verdicts (from the trace directly) ---------------------
# Over [0:50] s: min(y5-y4)=7.3894 while the other three gaps stay >7.5, so the
# overall verdict is SATISFIED iff the mutable threshold is below 7.3894.

@pytest.mark.parametrize(
    "thr,expected",
    [(7.0, V_SATISFIED), (7.5, V_VIOLATED), (8.0, V_VIOLATED)],
)
def test_ccx_handcomputed_verdicts(ccx_engine, ccx_formula, thr, expected):
    cand = _set(ccx_formula, {16: thr})
    verdict, _rho = ccx_engine.check(cand)
    assert verdict == expected


def test_ccx_seed_robustness_matches_gap_min(ccx_engine, ccx_formula):
    # Seed threshold 7.5 -> rho = min(y5-y4) - 7.5 = 7.3894 - 7.5.
    verdict, rho = ccx_engine.check(ccx_formula)
    assert verdict == V_VIOLATED
    assert rho == pytest.approx(7.3894 - 7.5, abs=1e-6)


def test_at1_seed_is_violated(at1_engine, at1_formula):
    # Ground truth from Z3: the seed AT1 property is VIOLATED.
    verdict, _rho = at1_engine.check(at1_formula)
    assert verdict == V_VIOLATED


# --- registry / subject resolution --------------------------------------------

def test_subject_from_path():
    assert subject_from_path(AT1_PATH) == "AT1"
    assert subject_from_path(CCX_PATH) == "CCX"


def test_unknown_subject_raises(tmp_path):
    prop = tmp_path / "ZZ9_x.py"
    prop.write_text("z3solver.add(timestamps[0]==0)\n")
    with pytest.raises(RtamtEngineError, match="no RTAMT template"):
        RtamtEngine(prop)


def test_registry_is_extensible():
    # Adding a subject is one registry entry: the engine reads templates from here.
    assert set(REGISTRY) == {"AT1", "CCX"}
    assert all(isinstance(v, Subject) for v in REGISTRY.values())
    assert REGISTRY["CCX"].positions == (16,)
    assert REGISTRY["AT1"].positions == (4, 8, 11)


def test_missing_rtamt_dependency_raises(monkeypatch):
    # Simulate the rtamt extra being absent -> a clear, actionable error.
    monkeypatch.setitem(__import__("sys").modules, "rtamt", None)
    with pytest.raises(RtamtUnavailable, match="pip install"):
        RtamtEngine(AT1_PATH, subject="AT1")
