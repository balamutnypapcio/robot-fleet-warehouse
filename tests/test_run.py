import csv
import subprocess
import sys
from pathlib import Path

import yaml

from sim.run import main

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "scenarios" / "default.yaml"


def small_scenario(tmp_path, **changes):
    """A fast batch scenario on the mini layout, written to tmp_path."""
    with open(DEFAULT, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data["layout"] = str(ROOT / "layouts" / "mini.yaml")
    data["mode"] = "batch"
    data["batch"] = {"orders": 3}
    data["policy"]["b_low"] = 40
    data.update(changes)
    path = tmp_path / "small.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_batch_run_prints_summary(tmp_path, capsys):
    assert main([str(small_scenario(tmp_path))]) == 0
    out = capsys.readouterr().out
    assert "mode:       batch (N = 3)" in out
    assert "3 delivered" in out
    assert "finished:   yes (s_fin)" in out


def test_out_writes_csv(tmp_path, capsys):
    log = tmp_path / "log.csv"
    assert main([str(small_scenario(tmp_path)), "--out", str(log)]) == 0
    with open(log, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["time", "event", "robot", "cell"]
    assert sum(r[1] == "NewOrder" for r in rows[1:]) == 3
    assert f"log:        {log}" in capsys.readouterr().out


def test_seed_override(tmp_path, capsys):
    path = small_scenario(tmp_path)
    main([str(path), "--seed", "42", "--out", str(tmp_path / "a.csv")])
    assert "seed:       42" in capsys.readouterr().out
    main([str(path), "--seed", "42", "--out", str(tmp_path / "b.csv")])
    main([str(path), "--seed", "43", "--out", str(tmp_path / "c.csv")])
    a, b, c = ((tmp_path / n).read_text() for n in ("a.csv", "b.csv", "c.csv"))
    assert a == b
    assert a != c


def test_module_runs_default_scenario_in_batch(tmp_path):
    """python -m sim.run works as a real command (the issue's CLI)."""
    with open(DEFAULT, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data["layout"] = str(ROOT / "layouts" / "warehouse_15x8.yaml")
    data["mode"] = "batch"
    data["batch"] = {"orders": 5}
    path = tmp_path / "batch.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "sim.run", str(path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "5 delivered" in proc.stdout
