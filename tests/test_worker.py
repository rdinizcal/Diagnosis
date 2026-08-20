from __future__ import annotations

import time
from pathlib import Path

import z3

from diagnosis.ga import replace_property_assertion
from diagnosis.harness import Verdict, run_property_script
from diagnosis.worker import (
    SolverWorker,
    V_ERROR,
    V_SATISFIED,
    V_UNDECIDED,
    V_VIOLATED,
    check_expr,
    make_eval_globals,
)


def _property_script(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "from z3 import *",
                "def tiny():",
                "\tz3solver = Solver()",
                "\tx = Int('x')",
                "\tz3solver.add(x >= 0)",
                "\tz3solver.add(Not(x >= 0))",
                "\tstatus = z3solver.check()",
                "\tprint(status)",
                "\tif status == sat:",
                "\t\tprint('REQUIREMENT VIOLATED')",
                "\telif status == unsat:",
                "\t\tprint('REQUIREMENT SATISFIED')",
                "\telse:",
                "\t\tprint('UNDECIDED')",
                "if __name__ == '__main__':",
                "\ttiny()",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_worker_protocol_round_trip(tmp_path):
    script = _property_script(tmp_path / "tiny.py")

    with SolverWorker(script) as worker:
        verdict, solve_seconds = worker.check("x < 0", timeout_ms=1000)

    assert verdict == V_SATISFIED
    assert solve_seconds >= 0.0


def test_worker_push_pop_isolation(tmp_path):
    script = _property_script(tmp_path / "tiny.py")

    with SolverWorker(script) as worker:
        first, _ = worker.check("x < 0", timeout_ms=1000)
        second, _ = worker.check("x == 1", timeout_ms=1000)
        third, _ = worker.check("x < 0", timeout_ms=1000)

    assert (first, second, third) == (V_SATISFIED, V_VIOLATED, V_SATISFIED)


def test_worker_sandbox_rejects_import(tmp_path):
    script = _property_script(tmp_path / "tiny.py")

    with SolverWorker(script) as worker:
        verdict, _ = worker.check("__import__('os').getcwd() or x < 0", timeout_ms=1000)

    assert verdict == V_ERROR


class _UnknownSolver:
    def push(self):
        pass

    def pop(self):
        pass

    def set(self, *args, **kwargs):
        pass

    def add(self, constraint):
        pass

    def check(self):
        return z3.unknown


def test_worker_timeout_maps_unknown_promptly():
    namespace = {"z3solver": _UnknownSolver(), "x": z3.Int("x")}
    start = time.perf_counter()

    verdict, _solve_seconds, _error = check_expr(
        namespace,
        make_eval_globals(namespace),
        "x > 0",
        timeout_ms=1,
    )

    assert verdict == V_UNDECIDED
    assert time.perf_counter() - start < 0.5


def test_worker_oracle_equivalence_with_subprocess(tmp_path):
    base_script = _property_script(tmp_path / "tiny.py")
    expressions = ["x < 0", "x == 1", "x >= 0"]
    expected = {
        Verdict.SAT: V_SATISFIED,
        Verdict.UNSAT: V_VIOLATED,
        Verdict.ERROR: V_ERROR,
    }

    with SolverWorker(base_script) as worker:
        for idx, expression in enumerate(expressions):
            candidate = tmp_path / f"candidate_{idx}.py"
            candidate.write_text(
                "".join(
                    replace_property_assertion(
                        base_script.read_text(encoding="utf-8").splitlines(keepends=True),
                        expression,
                    )
                ),
                encoding="utf-8",
            )
            subprocess_result = run_property_script(candidate, timeout=5)
            worker_verdict, _ = worker.check(expression, timeout_ms=5000)

            assert worker_verdict == expected[subprocess_result.verdict]
