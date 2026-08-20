"""Label-mode ARFF plumbing tests (Feature 2, mode="label").

Design decision (recorded here and in the report): adding an ``inferred``
attribute to the ARFF would make J48 treat it as a splittable feature and
perturb the tree, defeating the guide-vs-label tree comparison. So label mode
keeps the ARFF *schema* identical to guide/baseline and writes inferred rows as
normal data rows, recording the inferred provenance in a **sidecar CSV**
(``inferred_labels.csv``). These tests pin that behaviour.
"""

from __future__ import annotations

from pathlib import Path

from diagnosis.config import (
    HeuristicsConfig,
    IntervalInferenceConfig,
    TwoTierTimeoutConfig,
)
from diagnosis.inference import HeuristicsController
from diagnosis.lang.ast import IntConst, RelOp, Subscript, Var


def _seed():
    return RelOp("<", Subscript(base=Var("v_speed"), index=Var("i")), IntConst(120))


def _cfg(mode):
    return HeuristicsConfig(
        interval_inference=IntervalInferenceConfig(enabled=True, mode=mode, empirical_validation_k=1),
        two_tier_timeout=TwoTierTimeoutConfig(),
    )


def test_label_sidecar_written_with_inferred_flags(tmp_path):
    ctrl = HeuristicsController(_seed(), _cfg("label"), trace_check_timeout_sec=600)
    ctrl.record_arff_row("v_speed,<,120", "False", inferred=False)
    ctrl.record_arff_row("v_speed,<,150", "True", inferred=True)

    out = ctrl.write_sidecar(str(tmp_path))
    assert out is not None
    content = Path(out).read_text(encoding="utf-8").strip().splitlines()
    assert content[0] == "inferred,verdict,row"
    assert content[1] == "0,FALSE,v_speed,<,120"
    assert content[2] == "1,TRUE,v_speed,<,150"


def test_guide_mode_writes_no_sidecar(tmp_path):
    ctrl = HeuristicsController(_seed(), _cfg("guide"), trace_check_timeout_sec=600)
    # guide mode does not record sidecar rows...
    ctrl.record_arff_row("v_speed,<,150", "True", inferred=True)
    assert ctrl.inferred_sidecar == []
    assert ctrl.write_sidecar(str(tmp_path)) is None


def test_label_mode_includes_inferred_rows_in_dataset():
    """In label mode inferred candidates are kept for the ARFF (larger set)."""
    ctrl = HeuristicsController(_seed(), _cfg("label"), trace_check_timeout_sec=600)
    seed = _seed()
    cand140 = RelOp("<", Subscript(base=Var("v_speed"), index=Var("i")), IntConst(140))
    cand150 = RelOp("<", Subscript(base=Var("v_speed"), index=Var("i")), IntConst(150))
    p1 = ctrl.plan(cand140)
    ctrl.record_solve(p1, ["True"], "True")
    p2 = ctrl.plan(cand150)
    assert p2.inferred == "True"
    assert p2.include_in_arff is True


def test_arff_included_filters_inferred_at_write_time():
    """The ARFF writers exclude inferred rows at write time.

    A candidate can be appended to `self.unknown` (as Unknown, include=True)
    before evaluation and only *later* marked inferred in place, so the filter
    must run at write time, not at the append site. `_arff_included` is that
    choke point; with heuristics off it is the identity filter (parity).
    """
    from diagnosis.diagnostics.arff import _arff_included

    class Row:
        def __init__(self, include):
            self.include_in_arff = include

    real, inferred = Row(True), Row(False)
    assert _arff_included([real, inferred]) == [real]
    # objects lacking the attribute (baseline individuals) are always kept
    assert _arff_included([object()])  # non-empty


def test_guide_mode_excludes_inferred_rows_from_dataset():
    ctrl = HeuristicsController(_seed(), _cfg("guide"), trace_check_timeout_sec=600)
    cand140 = RelOp("<", Subscript(base=Var("v_speed"), index=Var("i")), IntConst(140))
    cand150 = RelOp("<", Subscript(base=Var("v_speed"), index=Var("i")), IntConst(150))
    p1 = ctrl.plan(cand140)
    ctrl.record_solve(p1, ["True"], "True")
    p2 = ctrl.plan(cand150)
    assert p2.inferred == "True"
    assert p2.include_in_arff is False
