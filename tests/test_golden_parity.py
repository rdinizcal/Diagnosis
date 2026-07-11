from __future__ import annotations

import datetime as _datetime
import json
import os
import re
import sys
from pathlib import Path

from diagnosis.config import load_config
from diagnosis.pipeline import run_diagnostics


REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = REPO_ROOT / "tests" / "golden"
FIXED_NOW = _datetime.datetime(2026, 1, 2, 3, 4, 5, 678901)


class _FixedDateTime(_datetime.datetime):
    @classmethod
    def now(cls, tz=None):
        if tz is None:
            return FIXED_NOW
        return tz.fromutc(FIXED_NOW.replace(tzinfo=tz))


def _weka_classpath() -> str | None:
    env_root = Path(sys.executable).resolve().parents[1]
    candidates = [
        env_root / "share" / "weka" / "weka-stable-3.8.6.jar",
        env_root / "share" / "weka" / "weka.jar",
    ]
    jars = [str(path) for path in candidates if path.exists()]
    bounce = env_root / "share" / "weka" / "bounce.jar"
    if bounce.exists():
        jars.append(str(bounce))
    return ":".join(jars) if jars else None


def _run_config(config_path: Path, output_dir: Path, monkeypatch) -> Path:
    import diagnosis.ga as ga_module

    monkeypatch.setattr(ga_module.datetime, "datetime", _FixedDateTime)
    classpath = _weka_classpath()
    if classpath:
        monkeypatch.setenv("WEKA_JAR", classpath)

    data = json.loads(config_path.read_text(encoding="utf-8"))
    data["input"]["output_dir"] = str(output_dir)
    run_config = output_dir / "config.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config.write_text(json.dumps(data, indent=2), encoding="utf-8")

    cfg = load_config(run_config)
    cwd = Path.cwd()
    os.chdir(REPO_ROOT)
    try:
        run_diagnostics(cfg)
    finally:
        os.chdir(cwd)

    runs = [path for path in output_dir.iterdir() if path.is_dir()]
    assert len(runs) == 1
    return runs[0]


def _verdict_sequence(run_dir: Path) -> list[str]:
    verdicts: list[str] = []
    # Per-generation population logs now live under generations/.
    for path in sorted(run_dir.glob("generations/[0-9][0-9].txt")):
        for line in path.read_text(encoding="utf-8").splitlines()[1:]:
            if line.startswith("HC: "):
                continue
            verdicts.append(line.rsplit("\t", 1)[-1])
    return verdicts


def _stable_tree_output(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if "core mtj jar files are not available as resources" in line:
            # Weka emits this JVM warning nondeterministically, and its text
            # embeds a JDK-version-specific classloader name plus a per-run
            # object hash. Drop it so the snapshot captures J48 behavior only.
            continue
        if line.startswith("Time taken "):
            label = line.split(":", 1)[0]
            lines.append(f"{label}: <elapsed>")
        else:
            line = re.sub(r"ARFF .*/(dataset_qty_[^/]+\.arff) has", r"ARFF <run>/\1 has", line)
            lines.append(line)
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def _snapshot(run_dir: Path) -> dict[str, object]:
    arff_path = next(run_dir.glob("dataset_qty_*_all.arff"))
    tree_path = run_dir / "J48-data-100.out"
    return {
        "dataset_all": arff_path.read_text(encoding="utf-8"),
        "verdict_sequence": _verdict_sequence(run_dir),
        "tree_100": _stable_tree_output(tree_path.read_text(encoding="utf-8")),
    }


def _golden_config_paths() -> list[Path]:
    return sorted(
        path for path in GOLDEN_DIR.glob("at1_seed*.json")
        if not path.name.endswith(".snapshot.json")
    )


def test_default_configs_match_golden_snapshots(tmp_path, monkeypatch):
    for config_path in _golden_config_paths():
        run_dir = _run_config(config_path, tmp_path / config_path.stem, monkeypatch)
        expected_path = GOLDEN_DIR / f"{config_path.stem}.snapshot.json"
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        assert _snapshot(run_dir) == expected
