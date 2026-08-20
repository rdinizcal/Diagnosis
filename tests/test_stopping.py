from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from diagnosis.diagnostics.j48 import run_j48
from diagnosis.diagnostics.summary import parse_j48_out
from diagnosis.ga import GA


def _ga_for_stopping(**overrides):
    ga = GA.__new__(GA)
    ga.stopping_config = SimpleNamespace(
        mode=overrides.get("mode", "cv_pr"),
        pr_threshold=overrides.get("pr_threshold", 0.95),
        check_every_generations=overrides.get("check_every_generations", 1),
        patience=overrides.get("patience", 1),
        min_samples=overrides.get("min_samples", 0),
        max_samples=overrides.get("max_samples", None),
    )
    ga.stopping_mode = ga.stopping_config.mode
    ga.stopping_checks = []
    ga._stopping_success_streak = 0
    ga._last_tree_hash = None
    ga._last_pr = overrides.get("_last_pr", None)
    ga._last_tree_stable = overrides.get("_last_tree_stable", None)
    ga.sats = overrides.get("sats", [])
    ga.unsats = overrides.get("unsats", [])
    ga.generation_counter = overrides.get("generation_counter", 0)
    return ga


def test_cv_pr_stopping_fires_on_synthetic_separable_metrics():
    ga = _ga_for_stopping(mode="cv_pr", pr_threshold=0.95, patience=2)

    assert ga._adaptive_decision_from_stats({"cv_precision": 0.98, "cv_recall": 0.97}) is False
    assert ga._adaptive_decision_from_stats({"cv_precision": 0.98, "cv_recall": 0.97}) is True


def test_one_class_guard_never_fires(monkeypatch):
    ga = _ga_for_stopping(sats=[object()], unsats=[])

    monkeypatch.setattr(
        "diagnosis.ga.run_j48",
        lambda *args, **kwargs: pytest.fail("one-class guard should not run J48"),
    )

    assert ga._check_adaptive_stopping() is False
    assert ga.stopping_checks == []


def test_parse_j48_weighted_cv_metrics_and_tree_hash():
    out_text = """
=== Classifier model (full training set) ===

J48 pruned tree
------------------
NUM0 <= 5: TRUE (3.0)
NUM0 > 5: FALSE (3.0)

Number of Leaves  : 	2
Size of the tree : 	3

=== Stratified cross-validation ===

=== Detailed Accuracy By Class ===

                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
Weighted Avg.    0.990    0.010    0.980      0.970    0.975      0.9      1.000     0.990
"""

    stats = parse_j48_out(out_text, include_stopping_metrics=True)

    assert stats["cv_precision"] == 0.98
    assert stats["cv_recall"] == 0.97
    assert stats["cv_f1"] == 0.975
    assert stats["tree_hash"] is not None


def test_default_parse_omits_stopping_metrics():
    # The report path must not gain adaptive-stopping keys when they are not
    # requested, so a default-config report stays byte-identical to legacy.
    stats = parse_j48_out("=== Classifier model ===\n: TRUE (3.0)\n")
    for key in ("tree_hash", "cv_precision", "cv_recall", "cv_f1"):
        assert key not in stats


def test_weighted_avg_with_question_marks_keeps_column_alignment():
    # Weka prints '?' for undefined precision/F cells; the reader must not let
    # a '?' shift the recall column onto precision.
    out_text = (
        "=== Stratified cross-validation ===\n"
        "=== Detailed Accuracy By Class ===\n"
        "                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC  ROC Area  PRC Area  Class\n"
        "Weighted Avg.    0.667    0.667    ?          0.667    ?          ?    0.500     0.556\n"
    )
    stats = parse_j48_out(out_text, include_stopping_metrics=True)
    assert stats["cv_precision"] is None
    assert stats["cv_recall"] == 0.667
    assert stats["cv_f1"] is None


def test_tree_stable_stops_on_repeated_hash():
    ga = _ga_for_stopping(mode="tree_stable", patience=1)
    assert ga._adaptive_decision_from_stats({"tree_hash": "abc"}) is False
    assert ga._adaptive_decision_from_stats({"tree_hash": "abc"}) is True
    # A changed tree resets the streak.
    assert ga._adaptive_decision_from_stats({"tree_hash": "xyz"}) is False


def test_cv_pr_stop_line_reports_precision_recall_instead_of_counts():
    ga = _ga_for_stopping(
        mode="cv_pr",
        pr_threshold=0.9,
        patience=3,
        _last_pr=(12, 0.91, 0.92, 2),
    )

    line = ga._stop_criteria_line(cum_sat=207, cum_unsat=191, target=1000)

    assert "precision=0.9100" in line
    assert "recall=0.9200" in line
    assert "sat 207/1000" not in line
    assert "unsat 191/1000" not in line


def test_cv_pr_plot_is_written(tmp_path):
    ga = _ga_for_stopping(mode="cv_pr", pr_threshold=0.9)
    ga.path = str(tmp_path)
    ga.stopping_checks = [
        {"generation": 1, "cv_precision": 0.70, "cv_recall": 0.65},
        {"generation": 2, "cv_precision": 0.85, "cv_recall": 0.82},
    ]

    ga._write_pr_plot()

    assert (tmp_path / "pr_growth.png").is_file()


def test_j48_output_is_captured_without_terminal_echo(monkeypatch, tmp_path, capsys):
    arff = tmp_path / "data.arff"
    arff.write_text(
        "\n".join([
            "@relation tiny",
            "@attribute X NUMERIC",
            "@attribute VEREDICT {TRUE,FALSE}",
            "@data",
            "1,TRUE",
        ]),
        encoding="utf-8",
    )

    class _Result:
        stdout = "weka model text\n"
        stderr = "weka warning text\n"

    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: _Result())

    out_path = run_j48(str(arff), 1.0, str(tmp_path))

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert "weka model text" in Path(out_path).read_text(encoding="utf-8")
    assert "weka warning text" in Path(out_path).read_text(encoding="utf-8")
