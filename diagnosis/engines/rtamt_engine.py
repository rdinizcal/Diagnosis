"""Lightweight RTAMT trace-checking engine (opt-in third checker).

This is a deliberately minimal alternative to the Z3 subprocess/worker engines,
used to demonstrate checker modularity: the GA, mutation, fitness, ARFF and J48
stages are untouched. The only new code lives here, at the checker-call boundary.
A mutated HLS formula is rendered into a hardcoded, per-subject STL template and
evaluated offline with RTAMT (discrete-time semantics) instead of Z3. The checker
stays a pure verdict oracle; no search heuristic touches this path.

Design notes
------------
* Template registry: :data:`REGISTRY` maps a requirement id to a :class:`Subject`.
  Adding a third subject (e.g. CC3) is one registry entry plus one ``build``
  function -- no engine changes.
* Time-unit conversion happens exactly once, in :meth:`RtamtEngine._us_to_sample`.
  Internal HLS time literals are microseconds; the ThEodorE encoding indexes the
  trace as ``ToInt(t / STEP_US)``, so a bound in microseconds maps to the discrete
  sample index ``floor(us / step_us)`` (floor matches Z3's ``ToInt`` truncation).
  RTAMT temporal bounds are therefore emitted as integer sample counts, which is
  exact and side-steps floating-point sampling-period rounding. Numeric (non-time)
  thresholds are passed through verbatim.
* Verdict mapping mirrors the tool's three-valued verdicts: robustness rho > 0 ->
  SATISFIED, rho < 0 -> VIOLATED, rho == 0 -> UNDECIDED (counted in ``rho_zero``;
  conservative, keeps strict/non-strict inequality ambiguity out of the dataset).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Tuple

from ..lang.ast import Formula
from ..lang.internal_encoder import _collect_positions

# Verdicts, matching the worker/subprocess engines.
V_SATISFIED = "SATISFIED"
V_VIOLATED = "VIOLATED"
V_UNDECIDED = "UNDECIDED"

# Internal HLS time literals are microseconds; seconds = microseconds / 1e6.
US_PER_SECOND = 1_000_000.0


class RtamtUnavailable(RuntimeError):
    """Raised when engine='rtamt' is selected but the rtamt extra is missing."""


class RtamtEngineError(RuntimeError):
    """Raised for trace/spec problems specific to the RTAMT engine."""


def _fmt(x: float) -> str:
    """Format a threshold/bound compactly and deterministically for STL text."""
    xf = float(x)
    return str(int(xf)) if xf.is_integer() else repr(xf)


def _build_at1(values: Dict[int, float], us_to_sample: Callable[[float], int]):
    """AT1: forall t in [lo, hi]: v_speed < thr  (window lo/hi + threshold)."""
    lo = us_to_sample(values[4])
    hi = us_to_sample(values[8])
    thr = values[11]
    if lo > hi:
        # Empty window -> the universally-quantified body is vacuously true.
        return None
    return f"out = always[{lo}:{hi}] (v_speed < {_fmt(thr)})"


def _build_ccx(values: Dict[int, float], us_to_sample: Callable[[float], int]):
    """CCX: always[0:50] over the four car-pair gaps; only y5-y4's bound mutates."""
    hi = us_to_sample(50_000_000.0)  # fixed 50 s window, snapped to samples
    thr = _fmt(values[16])
    return (
        f"out = (always[0:{hi}] (y5 - y4 > {thr})) "
        f"and (always[0:{hi}] (y4 - y3 > 7.5)) "
        f"and (always[0:{hi}] (y3 - y2 > 7.5)) "
        f"and (always[0:{hi}] (y2 - y1 > 7.5))"
    )


@dataclass(frozen=True)
class Subject:
    """One requirement's STL template and the AST positions it reads."""
    name: str
    signals: Tuple[str, ...]          # HLS signals -> trace columns (by name)
    positions: Tuple[int, ...]        # mutable position ids filled into the template
    build: Callable[[Dict[int, float], Callable[[float], int]], "str | None"]


# Per-subject registry. CC3 slots in here as one more entry when the authors decide.
REGISTRY: Dict[str, Subject] = {
    "AT1": Subject("AT1", ("v_speed",), (4, 8, 11), _build_at1),
    "CCX": Subject("CCX", ("y1", "y2", "y3", "y4", "y5"), (16,), _build_ccx),
}


def subject_from_path(property_path: str | Path) -> str:
    """Resolve the requirement id from the property file name (prefix before '_')."""
    return Path(property_path).stem.split("_")[0].upper()


_ASSIGN = re.compile(r"(\w+)\[\s*(\d+)\s*\]\s*==\s*([-+0-9.eE]+)")


def _load_trace(
    property_path: str | Path, signals: Tuple[str, ...]
) -> Tuple[Dict[str, list], float, float, int]:
    """Load the inlined trace once from the (authoritative) property script.

    Returns ``(arrays, step_us, period_s, n)``. The property file is the exact
    trace Z3 consumes, so verdicts stay comparable across engines.
    """
    ts: Dict[int, float] = {}
    sig: Dict[str, Dict[int, float]] = {s: {} for s in signals}
    for line in Path(property_path).read_text(encoding="utf-8").splitlines():
        m = _ASSIGN.search(line)
        if not m:
            continue
        name, idx, val = m.group(1), int(m.group(2)), float(m.group(3))
        if name == "timestamps":
            ts[idx] = val
        elif name in sig:
            sig[name][idx] = val
    if len(ts) < 2:
        raise RtamtEngineError(f"no timestamp trace found in {property_path}")
    n = len(ts)
    times = [ts[i] for i in range(n)]
    steps = {round(times[i + 1] - times[i], 6) for i in range(n - 1)}
    if len(steps) != 1:
        raise RtamtEngineError(
            f"non-constant sampling period in {property_path}: {sorted(steps)}"
        )
    step_us = times[1] - times[0]
    if step_us <= 0:
        raise RtamtEngineError(f"non-positive sampling period in {property_path}")
    arrays = {s: [sig[s][i] for i in range(n)] for s in signals}
    return arrays, step_us, step_us / US_PER_SECOND, n


class RtamtEngine:
    """Render mutated candidates into STL and evaluate them with RTAMT offline."""

    def __init__(self, property_path: str | Path, subject: str | None = None) -> None:
        try:
            import rtamt  # noqa: F401
        except ImportError as exc:  # pragma: no cover - exercised via message test
            raise RtamtUnavailable(
                "engine='rtamt' requires the optional 'rtamt' dependency. "
                "Install it with: pip install 'diagnosis[rtamt]'"
            ) from exc

        self.subject_id = (subject or subject_from_path(property_path)).upper()
        if self.subject_id not in REGISTRY:
            raise RtamtEngineError(
                f"no RTAMT template for subject {self.subject_id!r}; "
                f"known subjects: {sorted(REGISTRY)}"
            )
        self.subject = REGISTRY[self.subject_id]
        self.signals, self.step_us, self.period_s, self.n = _load_trace(
            property_path, self.subject.signals
        )
        self._time = list(range(self.n))
        self._spec_cache: Dict[str, object] = {}
        self.rho_zero = 0
        self.evaluations = 0

    def _us_to_sample(self, us: float) -> int:
        """Convert an internal microsecond bound to a discrete sample index.

        The single time-unit conversion: microseconds -> seconds (/1e6) -> sample
        index (/period_s), floored to match Z3's ``ToInt`` truncation.
        """
        return math.floor(float(us) / self.step_us)

    def render(self, candidate_ast: Formula):
        """Render one candidate AST into its STL spec string (or None if vacuous)."""
        positions = _collect_positions(candidate_ast)
        values = {i: float(positions[i].node.value) for i in self.subject.positions}
        return self.subject.build(values, self._us_to_sample)

    def _robustness(self, spec_str: str) -> float:
        spec = self._spec_cache.get(spec_str)
        if spec is None:
            import rtamt

            spec = rtamt.StlDiscreteTimeSpecification()
            for name in self.subject.signals:
                spec.declare_var(name, "float")
            spec.spec = spec_str
            spec.parse()
            self._spec_cache[spec_str] = spec
        dataset = {"time": self._time}
        dataset.update(self.signals)
        rob = spec.evaluate(dataset)
        return float(rob[0][1])

    def check(self, candidate_ast: Formula) -> Tuple[str, float]:
        """Return ``(verdict, robustness)`` for one candidate formula AST."""
        self.evaluations += 1
        spec_str = self.render(candidate_ast)
        if spec_str is None:
            return V_SATISFIED, float("inf")
        rho = self._robustness(spec_str)
        if rho > 0:
            return V_SATISFIED, rho
        if rho < 0:
            return V_VIOLATED, rho
        self.rho_zero += 1
        return V_UNDECIDED, rho
