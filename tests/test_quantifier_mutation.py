from __future__ import annotations

import random

from diagnosis.lang.ast import And, BoolConst, Exists, ForAll, Implies, IntConst, RelOp, Var
from diagnosis.mutation.api import MutationConfig, mutate_formula


def _quantifier_only() -> MutationConfig:
    return MutationConfig(
        max_mutations=1,
        allowed_positions={0},
        enable_numeric_perturbation=False,
        enable_relop_flip=False,
        enable_logical_flip=False,
        enable_quantifier_flip=True,
    )


def _range_guard(variable: str = "t") -> And:
    return And(
        args=[
            RelOp(op="<=", left=IntConst(0), right=Var(variable)),
            RelOp(op="<=", left=Var(variable), right=IntConst(10)),
        ]
    )


def test_forall_guard_becomes_existential_conjunction():
    guard = _range_guard()
    payload = RelOp(op="<", left=Var("speed"), right=IntConst(120))
    original = ForAll(vars=["t"], body=Implies(left=guard, right=payload))

    mutated = mutate_formula(original, _quantifier_only(), random.Random(0))

    assert mutated == Exists(vars=["t"], body=And(args=[guard, payload]))


def test_exists_guard_becomes_universal_implication_and_keeps_payload():
    guard = _range_guard()
    payload = And(args=[BoolConst(True), BoolConst(False)])
    original = Exists(vars=["t"], body=And(args=[guard, *payload.args]))

    mutated = mutate_formula(original, _quantifier_only(), random.Random(0))

    assert mutated == ForAll(vars=["t"], body=Implies(left=guard, right=payload))


def test_unguarded_quantifier_is_not_flipped():
    original = ForAll(
        vars=["t"],
        body=RelOp(op="<", left=Var("speed"), right=IntConst(120)),
    )

    assert mutate_formula(original, _quantifier_only(), random.Random(0)) == original


def test_guard_must_reference_the_bound_variable():
    unrelated_guard = _range_guard("u")
    original = ForAll(
        vars=["t"],
        body=Implies(left=unrelated_guard, right=BoolConst(True)),
    )

    assert mutate_formula(original, _quantifier_only(), random.Random(0)) == original


def test_exists_requires_a_relational_range_guard():
    original = Exists(
        vars=["t"],
        body=And(args=[BoolConst(True), RelOp(op="<", left=Var("t"), right=IntConst(5))]),
    )

    assert mutate_formula(original, _quantifier_only(), random.Random(0)) == original
