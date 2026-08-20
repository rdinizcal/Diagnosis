# Diagnosis
`diagnosis` is a research tool for **diagnosing falsified requirements** in HLS/ThEodorE-based workflows using **search-based mutation** and **decision-tree learning**.

Given:

- a **falsified requirement** (a ThEodorE-generated Python property file),
- a set of **execution traces** (encoded as Z3 constraints),
- and a configuration describing how formulas may **mutate**,

`diagnosis`:

1. Parses the requirement into a **typed AST**.
2. Uses a **genetic algorithm (GA)** over structured mutations of that AST.
3. Evaluates each mutated requirement against the traces via **Z3**.
4. Logs the resulting verdicts and features into an **ARFF dataset**.
5. Runs **Weka J48** to learn a **decision tree** that explains why the original requirement failed.

The result is a **diagnostic decision tree** that relates parts of the requirement to satisfiability/violation patterns over execution traces.

---

## Research artifact

The paper's data, archived runs, experiment configurations, and reproduction
scripts live in the separate
[`sbtd-replication-package`](https://github.com/rdinizcal/sbtd-replication-package)
repository. This repository contains only the Diagnosis tool, examples, and
tests. The artifact pins the exact historical Diagnosis snapshot it uses as a
Git submodule; do not substitute the latest release when reproducing archived
results.

---

## Package layout (`diagnosis`)

```text
diagnosis/
├── cli.py                   # CLI: run, explain-positions, inspect-internal, ...
├── config.py                # JSON config -> dataclasses
├── pipeline.py              # Unified pipeline: property → AST → GA → ARFF → J48
│
├── lang/
│   ├── ast.py               # Typed AST definitions
│   ├── theodore_parser.py   # Parse ThEodorE Python property → AST
│   ├── internal_encoder.py  # AST → GA internal format + ARFF layout
│   ├── python_printer.py    # AST → executable Python script for Z3
│   └── analysis.py          # AST utilities (size, vars, depth, etc.)
│
├── mutation/
│   └── api.py               # MutationConfig + mutate_formula (typed, safe)
│
├── diagnostics/
│   ├── arff.py              # ARFF writer
│   └── j48.py               # Weka J48 runner
│
├── ga.py                    # GA loop
├── individual.py            # GA individual (AST + internal treenode)
├── harness.py               # Safe Z3 evaluation
└── examples/                # Example ThEodorE properties
```

---  

## What the tool does (high-level)

1. Read a ThEodorE Python property (z3solver.add(Not(ForAll(...)))).
2. Parse to a typed AST (lang.ast).
3. Encode AST → internal GA format + ARFF layout.
4. Mutate formulas through typed AST operators (numeric jitter, flips, bounds).
5. Evaluate mutated properties via the safe Z3 harness (SAT / UNSAT / ERROR).
6. Generate ARFF datasets from features extracted during encoding.
7. Learn decision trees via Weka J48.

All of this orchestrated through:
```
diagnosis run --config <file.json>
```

---

## Prerequisites

You can use the docker installation or local cli installation:

For Docker (recommended):
- Docker 

Build docker. From /docker run:
```
docker compose build
```

Check CLI availability:
```
docker compose run --rm diagnosis --help
docker compose run --rm diagnosis --version
```

If you do not want to use the docker installation, you'll need:
- Python ≥ 3.8
- Z3 (pip install z3-solver)
- Java (for using Weka J48)

Install the python envrionment:
```
python -m pip install -e .
```

Simply use the tool via:
```
diagnosis --help
```

---

## Usage (docker-oriented)

Basic command help:
```
docker compose run --rm diagnosis --help
```

Full pipeline (AT1 example):
```
docker compose run --rm  diagnosis run --config configs/AT1_config_exp1.json
```

This performs:
```
ThEodorE Python property
        ↓
   AST (lang.ast)
        ↓ encode
Internal GA format + ARFF layout
        ↓
      GA loop
  (AST-based mutation)
        ↓ evaluate
     Z3 harness
        ↓
  ARFF dataset(s)
        ↓ 
    Weka J48
        ↓
 Diagnostic tree
```

All output is written in the output directory.

---

## Configuration Overview

A config JSON includes:
- input.requirement_file – Python property file from ThEodorE
- input.traces_file – trace Z3 encoding
- ga.* – population, generations, seed, rates (ga.stopping.* – optional adaptive stopping, see "Performance options")
- mutation.* – allowed positions, numeric bounds, operator toggles
- diagnostics.* – ARFF output, Weka path, J48 options
- evaluation.* – optional evaluation tuning (cache, worker engine, parallelism, timeout), see "Performance options"

Example:
```json
{
  "input": {
    "requirement_file": "diagnosis/examples/AT1_AT001.py",   // ThEodorE-generated Python requirement
    "traces_file": "diagnosis/tracesAT.csv",                // Execution traces encoded for Z3
    "output_dir": "outputs"                              // Directory for ARFF + decision trees + logs
  },

  "ga": {
    "population_size": 50,       // Number of candidate formulas per generation
    "generations": 10,           // Number of evolution iterations
    "crossover_rate": 0.95,      // Probability of combining two parents
    "mutation_rate": 0.10,       // Probability of mutating an individual
    "seed": 42,                  // Random seed for reproducibility
    "target_sats": 10,           // Target number of formaulas to be satisfied and unsatisfied (stop criterion)

    // Optional adaptive stopping (see "Performance options"). Omitting this
    // block, or leaving mode="count", reproduces the legacy stop criterion.
    "stopping": {
      "mode": "count"            // "count" (default) | "cv_pr" | "tree_stable"
    }
  },

  // Optional evaluation tuning (see "Performance options"). Every key is
  // opt-in; omitting the whole block preserves legacy behaviour.
  "evaluation": {
    "trace_check_timeout_sec": 3600,  // Per-candidate Z3 trace-check timeout (legacy hard-coded value)
    "cache_enabled": false,           // Memoize verdicts by property-expression hash
    "engine": "subprocess",           // "subprocess" (default) | "worker" | "rtamt"
    "parallel_workers": 1             // Evaluate a generation's candidates over N workers (1 = serial)
  },

  "mutation": {
    "max_mutations": 1,          // Max number of AST-level edits per mutation step

    "enable_numeric_perturbation": true,  // Allow jittering numeric constants
    "enable_relop_flip": true,            // Allow flipping relational operators (< → >, ≤ → ≥, etc.)
    "enable_logical_flip": true,          // Allow switching AND ↔ OR ↔ Implies
    "enable_quantifier_flip": false,      // Allow switching ∀ ↔ ∃

    // Specific AST positions that are allowed to mutate (see explain-positions)
    "allowed_positions": [2,6,11],

    // Fine-grained control over *what* each position may change into
    "allowed_changes": {

      // Position 2: logical connective restricted to And, Or only, excludes Implies
      "2": {
        "logical": ["And","Or"]
      },

      // Position 6: relational operator restricted to < or >
      "6": {
        "relational": ["<", ">"]
      },

      // Position 11: numeric literal bounded to [100.0, 140.0]
      "11": {
        "numeric": [100.0, 140.0]
      }
  }
}

```

---

## Performance options

These keys tune *how* candidates are evaluated and *when* evolution stops.
They are performance optimizations, not behavior changes: **every option
defaults to a value that reproduces legacy behavior bit-for-bit.** A config
that omits the `evaluation` block and leaves `ga.stopping.mode` at `"count"`
produces the same verdicts, GA trajectory, ARFF datasets, J48 trees, and
`report.json` (modulo wall-clock/resource fields) as before these options
existed. Speedups are workload-dependent; enable an option and compare the
wall-clock fields in `report.json` for your requirement.

### `evaluation` block

| Key | Default | Effect |
| --- | --- | --- |
| `trace_check_timeout_sec` | `3600` | Per-candidate Z3 trace-check timeout, in seconds. The default matches the previously hard-coded one-hour limit, so it changes nothing; lower it to fail slow/undecidable candidates faster. |
| `cache_enabled` | `false` | When `true`, verdicts are memoized by `sha256` of the property-expression string in an in-memory dict backed by a SQLite file in the run directory, and consulted before any solver call. Verdict-preserving by construction (a cache hit returns the verdict the solver already produced). Hit/miss/distinct counters are written to `report.json` only when enabled. Helps workloads where formulas recur across generations. |
| `engine` | `"subprocess"` | `"subprocess"` spawns a fresh Python+Z3 process per candidate. `"worker"` keeps a long-lived Z3 process. `"rtamt"` selects the optional RTAMT oracle for the supported AT1 and CCX templates and requires serial execution with search heuristics disabled. On Python 3.8–3.12, install it with `pip install -e '.[rtamt]'`; RTAMT 0.3.5's ANTLR dependency is not compatible with Python 3.13+. |
| `parallel_workers` | `1` | Number of candidates from one generation to evaluate concurrently, each with its own temp file (the shared temp file is used only on the serial path). Results are applied in population-index order after the batch completes, so datasets and `report.json` are independent of worker count and scheduling. `1` keeps the serial path untouched. The cache (if enabled) is shared race-safely via SQLite WAL. |

### `ga.stopping` block

| Key | Default | Effect |
| --- | --- | --- |
| `mode` | `"count"` | `"count"` keeps the legacy stop criterion (at least `target_sats` sat *and* `target_sats` unsat samples). `"cv_pr"` stops once J48 weighted cross-validated precision **and** recall are both ≥ `pr_threshold` for `patience` consecutive checks. `"tree_stable"` stops once the normalized root+depth-2 tree hash is unchanged for `patience` consecutive checks. |
| `pr_threshold` | `0.95` | Precision/recall bar for `cv_pr`. |
| `check_every_generations` | `1` | Minimum number of generations between checks. |
| `patience` | `2` | Number of consecutive passing checks required before stopping. |
| `min_samples` | `0` | No check runs until this many cumulative samples exist. |
| `max_samples` | `null` | Hard cap; evolution stops as soon as the cumulative sample count reaches it. |

Any adaptive check additionally requires at least one sat **and** one unsat
sample before it can fire (one-class guard), and each check is recorded in the
run summary under `ga_stopping_checks` when a non-`count` mode is active.

### `heuristics` block (search-pruning heuristics)

Four opt-in heuristics cut solver work without losing verdicts. **All default
OFF**; a config with no `heuristics` block is bit-identical to the baseline
(verified by `tests/test_golden_parity.py`). They are gated by a static
monotonicity proof and protected by a runtime consistency guard.

```jsonc
"heuristics": {
  "interval_inference": {
    "enabled": false,
    "mode": "guide",              // "guide" | "label"
    "empirical_validation_k": 3,  // confirming solves before a direction is trusted
    "min_gap": 1e-6               // relative bracket width below which shrinking stops
  },
  "two_tier_timeout": {
    "enabled": false,
    "low_sec": 60,
    "high_sec": 600,
    "escalation": "once_per_formula"
  },
  "adaptive_range": {
    "enabled": false,
    "exploration_fraction": 0.15,
    "endpoint_init": true,
    "on_one_class": "report_and_stop",
    "widen_factor": 1.5,
    "max_widenings": 4
  },
  "time_quantization": {
    "enabled": false,
    "validate_every_n_hits": 50,  // periodic same-class double-solve validation
    "period": null,               // optional override; must match the trace spacing
    "force_period": false         // validation-only: skip the period cross-check
  }
}
```

**`interval_inference`** — for a candidate that differs from the seed only at
numeric positions whose polarity is provably monotone (see `explain-polarity`),
the trace-check verdict is monotone in those constants. The tool reasons over the
**joint** space of all monotone knobs (parametric-STL monotonicity is
per-coordinate, so SATISFIED is an up-set and VIOLATED a down-set in the product
/ Pareto order) and *infers* a candidate's verdict with **no solver call** when it
**dominates** a known-SATISFIED point (⇒ SATISFIED) or is **dominated by** a
known-VIOLATED point (⇒ VIOLATED). With a single monotone knob this reduces to
the classic SATISFIED / VIOLATED half-lines plus UNDECIDED band; with several
knobs it infers candidates that moved *all* of them at once. A candidate that
also changes a **non-monotone** (`UNKNOWN`-polarity) numeric knob, or changes the
formula structurally, is always solved.

Soundness is doubly enforced: (1) inference is attempted only on
statically-proven monotone knobs, and (2) after **every** real solve the verdict
is checked against the region — a SATISFIED point dominated by a VIOLATED point
(or vice versa) is a `monotonicity_violation`, which permanently disables
inference, discards the region, and records a witness in `report.json`. In
addition, `empirical_validation_k` consistent real solves are required before any
verdict is inferred.

- `mode: "guide"` — inferred verdicts only skip solver calls for the GA's
  bookkeeping; inferred individuals are **excluded** from the ARFF dataset (the
  decision tree trains on real verdicts only). Zero effect on diagnosis
  validity; smaller training set. **Default.**
- `mode: "label"` — inferred verdicts are written into the ARFF as normal rows
  (larger training set). To keep the J48 tree comparable to guide/baseline, the
  inferred flag is **not** added as an ARFF attribute (J48 would split on it);
  instead it is recorded in an `inferred_labels.csv` sidecar in the run
  directory. Requires the soundness argument in the paper.
**`two_tier_timeout`** — solve with `low_sec` first; on UNDECIDED, re-solve once
with `high_sec` (`report.json` counters: `tier1_decided`, `tier2_decided`,
`tier2_undecided`). A formula whose high-tier solve is UNDECIDED is cached as
UNDECIDED, so exact repeats never re-solve. **Region memory:** when interval
inference is also on and a single-mutation candidate lands inside a *confirmed*
UNDECIDED band, the high tier is skipped (`region_memory_skips`) — slow
requirements whose unknowns burn any budget stop paying repeatedly.
`high_sec` must be `<= evaluation.trace_check_timeout_sec` (validated at load).

**`adaptive_range`** — changes numeric candidate generation for a monotone
position. With `endpoint_init: true`, the two configured range endpoints are
evaluated before generation 0. If the endpoints are the same decisive class,
the run records a structured `one_class_space` finding in `report.json` and
`summary.json`; `on_one_class` controls whether the run stops, widens the range
in the polarity-easier direction, or continues for comparison. During mutation,
draws come from the current SAT/UNSAT unresolved bracket with probability
`1 - exploration_fraction`, and from the full configured range otherwise. If no
bracket exists, a confirmed UNDECIDED band is reached, or the monotonicity guard
disables inference for that position, sampling reverts to the full range.

This is a **search-behavior change**: results obtained with
`adaptive_range.enabled: true` require re-validation against expert ground
truth because candidate distribution and GA trajectory change.

**`time_quantization`** — traces are sampled at a fixed period (detected from the
`ToInt(… / PERIOD)` index and cross-checked against the trace timestamp spacing),
and a signal is read as `v_speed[ToInt((t - offset) / PERIOD)]`. A mutated
time-window bound `B` therefore changes the verdict only through
`floor(B / PERIOD)`: every `B` inside one inter-sample interval yields an
equivalent formula. When ON, the verdict cache is consulted with a **canonical**
key in which every quantizable time token is replaced by its class index
`floor(value/period)`, so distinct raw bounds in one class collapse onto a single
solve. **Only the key changes** — the formula sent to the solver on a miss is the
original, un-canonicalized one, and the feature composes with both engines and the
two-tier timeout unchanged. Which positions are quantizable (and their period) is
shown by `explain-polarity`.

Two safety nets back the static gate: every `validate_every_n_hits` same-class
cache hits, the new value is actually re-solved and compared; a mismatch disables
quantization for that position, purges the class's cache entries, and logs a
`quantization_violation` witness (must be 0 in a valid run). A **vacuity guard**
flags a candidate whose two mutable bounds define an empty window (`a >= b`, a
vacuously SATISFIED `ForAll`) so it never feeds the tree as an ordinary SAT.
`report.json` gains `quantized_positions`, `quant_hits_exact` /
`quant_hits_canonical`, `quant_validations`, `quantization_violations`, and
`vacuous_candidates`.

**In combination with `adaptive_range`**, a quantizable monotone bound brackets
on **class indices** rather than raw floats (highest-known-UNSAT class to
lowest-known-SAT class), with a minimum gap of one sample period; the bracket
bisects to the boundary in `ceil(log2(#classes))` real solves. The recovered
breakpoint is recorded in `report.json` as `boundary_at` — the **sample
timestamp** between the two adjacent classes. Reported time boundaries are
therefore sample-aligned: **this is more faithful to the trace, not less
precise**, since sub-sample bounds are indistinguishable to the encoding.

This is a **search-behavior change**: results obtained with
`time_quantization.enabled: true` require re-validation against expert ground
truth, and any reported boundary must be read as sample-aligned.

Per-run region state is persisted to `inference_state.json`; heuristic counters
are folded into `report.json`.

---

## Inspecting monotonicity (`explain-polarity`)

To see the proven direction of every numeric threshold position (this gates
interval inference):
```
diagnosis explain-polarity --config configs/AT1_config_exp1.json
```
```
Pos  Direction    Quantizable           Reason
----------------------------------------------------------------------------------------------------
 11  INCREASING   no                    threshold right of '<' against a signal term; polarity flag preserved (+) -> INCREASING
```
`UNKNOWN` means inference is disabled for that position (equality operators,
constants under arithmetic, non-quantizable temporal-window bounds, or any path
the analysis cannot soundly classify). The **Quantizable** column reports whether
a position is a sample-aligned time bound and its detected period; such bounds
are promoted from `UNKNOWN` to a definite direction (the floor of a monotone
index is monotone) and drive the `time_quantization` heuristic.

---

## Helper Commands for Introspection (AST ↔ positions ↔ ARFF)

To see where every mutation position sits in the requirement:
```
docker compose run --rm  diagnosis explain-positions --config configs/AT1_config_exp1.json
```

Example:
```
Pos  NodeType            Node                                      Features
--------------------------------------------------------------------------------
  0  ForAll              ∀ t. (((0 <= t) ∧ (t <= 20000000)) → ...  QUANTIFIERS0
  1  Implies             (((0 <= t) ∧ (t <= 20000000)) → (v_sp...  LOGICALS0
  2  And                 ((0 <= t) ∧ (t <= 20000000))              LOGICALS1
  3  RelOp               (0 <= t)                                  RELATIONALS0
  4  IntConst            0                                         NUM0
  5  Var                 t                                         TERM0
  6  RelOp               (t <= 20000000)                           RELATIONALS1
  7  Var                 t                                         TERM1
  8  IntConst            20000000                                  NUM1
  9  RelOp               (v_speed[ToInt((RealVal(0) + ((t - 0...   RELATIONALS2
 10  Subscript           v_speed[ToInt((RealVal(0) + ((t - 0 ...   TERM2
 11  RealConst           120.0                                     NUM2

```


To inspect a single mutation slot:
```
docker compose run --rm  diagnosis explain-position --config configs/AT1_config_exp1.json --position 12
```

---

Happy diagnosing!
