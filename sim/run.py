"""Run one scenario from the command line and print a summary.

    python -m sim.run scenarios/default.yaml
    python -m sim.run scenarios/default.yaml --seed 7 --out log.csv

--seed overrides the seed of the scenario file; --out writes the event log as CSV
(see sim.simulator.write_csv). The exit code is 1 if a batch run does not reach s_fin.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from collections import Counter
from pathlib import Path

from fleet.events import NewOrder
from fleet.state import Order
from sim.scenario import Scenario, load_scenario, run_scenario
from sim.simulator import Result, write_csv


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m sim.run", description="Simulate one warehouse scenario."
    )
    parser.add_argument("scenario", type=Path, help="scenario YAML file")
    parser.add_argument("--seed", type=int, help="override the seed of the scenario file")
    parser.add_argument("--out", type=Path, help="write the event log to this CSV file")
    return parser.parse_args(argv)


def summary(sc: Scenario, result: Result, orders: list[Order], wall: float) -> str:
    """Human-readable report of one run."""
    events = Counter(type(e.event).__name__ for e in result.log)
    s = result.state
    if sc.mode == "continuous":
        mode = f"continuous (lambda = {sc.rate:g} /s, duration = {sc.duration:g} s)"
    else:
        mode = f"batch (N = {sc.orders})"
    lines = [
        f"scenario:   {sc.name or '-'}  [{sc.layout.name}]",
        f"mode:       {mode}",
        f"seed:       {sc.seed}",
        f"robots:     {len(s.robots)}",
        f"orders:     {len(orders)} generated, {events[NewOrder.__name__]} arrived, "
        f"{s.delivered} delivered, {len(s.queue)} in queue",
    ]
    if sc.mode == "batch":
        lines.append(f"finished:   {'yes (s_fin)' if result.finished else 'NO (stuck)'}")
    lines += [
        f"sim time:   {result.time:.1f} s",
        f"wall time:  {wall:.2f} s",
        f"events:     {len(result.log)}  "
        + ", ".join(f"{name}={n}" for name, n in sorted(events.items())),
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sc = load_scenario(args.scenario)
    if args.seed is not None:
        sc = dataclasses.replace(sc, seed=args.seed)
    t0 = time.perf_counter()
    result, orders = run_scenario(sc)
    wall = time.perf_counter() - t0

    print(summary(sc, result, orders, wall))
    if args.out is not None:
        write_csv(result.log, args.out)
        print(f"log:        {args.out}")
    # a batch run that does not reach s_fin is a failure (e.g. for scripts and CI)
    return 1 if sc.mode == "batch" and not result.finished else 0


if __name__ == "__main__":
    sys.exit(main())
