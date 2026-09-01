"""Unit tests for the past-time temporal operator AST nodes.

Covers the string rendering of H, O, Y, Z and S, their nesting with the
existing propositional nodes, and the Formula/frozen invariants shared with
the rest of the AST hierarchy.
"""

from __future__ import annotations

import dataclasses

import pytest

from diagnosis.lang.ast import (
    Formula,
    Historically,
    Implies,
    Once,
    Since,
    Var,
    WeakYesterday,
    Yesterday,
)

UNARY_NODES = [Historically, Once, Yesterday, WeakYesterday]


def test_historically_str():
    assert str(Historically(Var("p"))) == "H(p)"


def test_once_str():
    assert str(Once(Var("p"))) == "O(p)"


def test_yesterday_str():
    assert str(Yesterday(Var("p"))) == "Y(p)"


def test_weak_yesterday_str():
    assert str(WeakYesterday(Var("p"))) == "Z(p)"


def test_since_str():
    assert str(Since(Var("p"), Var("q"))) == "(p S q)"


def test_nested_temporal_str():
    node = Historically(Implies(Yesterday(Var("p")), Var("q")))
    assert str(node) == "H((Y(p) → q))"


def test_nested_since_str():
    node = Once(Since(Historically(Var("p")), WeakYesterday(Var("q"))))
    assert str(node) == "O((H(p) S Z(q)))"


@pytest.mark.parametrize("cls", UNARY_NODES)
def test_unary_nodes_are_formulas(cls):
    assert isinstance(cls(Var("p")), Formula)


def test_since_is_a_formula():
    assert isinstance(Since(Var("p"), Var("q")), Formula)


@pytest.mark.parametrize("cls", UNARY_NODES)
def test_unary_nodes_are_frozen(cls):
    node = cls(Var("p"))
    with pytest.raises(dataclasses.FrozenInstanceError):
        node.body = Var("q")


def test_since_is_frozen():
    node = Since(Var("p"), Var("q"))
    with pytest.raises(dataclasses.FrozenInstanceError):
        node.left = Var("r")
