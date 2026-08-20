from __future__ import annotations

import pytest

from diagnosis.ga import replace_property_assertion


def test_replaces_not_implies_property_without_touching_trace_constraints():
    lines = [
        "from z3 import *\n",
        "def property():\n",
        "    z3solver = Solver()\n",
        "    z3solver.add(trace_a[0] >= 0)\n",
        "    z3solver.add(trace_b[0] <= 1)\n",
        "    z3solver.add(Not(Implies(ForAll([t], trace_a[t] > 0), ForAll([t], trace_b[t] < 1))))\n",
        "    print(z3solver.check())\n",
    ]

    result = replace_property_assertion(lines, "Not(Exists([t], trace_a[t] < 0))")

    assert result[3] == lines[3]
    assert result[4] == lines[4]
    assert result[5] == "    z3solver.add(Not(Exists([t], trace_a[t] < 0)))\n"
    assert "Implies(ForAll" not in "".join(result)


def test_single_quantifier_property_replacement_is_byte_identical():
    lines = [
        "from z3 import *\n",
        "def property():\n",
        "\tz3solver = Solver()\n",
        "\tz3solver.add(speed[0] >= 0)\n",
        "\tz3solver.add(Not(ForAll([t], Implies(t >= 0, speed[t] <= 120))))\n",
        "\tprint(z3solver.check())\n",
    ]

    result = replace_property_assertion(lines, "Not(ForAll([t], speed[t] <= 130))")

    assert result == [
        "from z3 import *\n",
        "def property():\n",
        "\tz3solver = Solver()\n",
        "\tz3solver.add(speed[0] >= 0)\n",
        "\tz3solver.add(Not(ForAll([t], speed[t] <= 130)))\n",
        "\tprint(z3solver.check())\n",
    ]


def test_property_replacement_rejects_missing_marker():
    with pytest.raises(RuntimeError, match="Expected exactly one"):
        replace_property_assertion(["z3solver.add(trace[0] >= 0)\n"], "Not(True)")
