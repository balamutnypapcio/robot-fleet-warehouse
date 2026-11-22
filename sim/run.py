"""Run one scenario from the command line and print a summary.

    python -m sim.run scenarios/default.yaml
    python -m sim.run scenarios/default.yaml --seed 7 --out log.csv
    python -m sim.run scenarios/default.yaml --plot-orders [orders.png]

--seed overrides the seed of the scenario file; --out writes the event log as CSV
(see sim.simulator.write_csv); --plot-orders plots the distribution of the generated orders
(see sim.orders). The exit code is 1 if a batch run does not reach s_fin.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from fleet.events import NewOrder
from fleet.layout import load_layout
from fleet.state import Order
from sim.orders import plot_orders
from sim.scenario import Scenario, load_scenario, run_scenario
from sim.simulator import Result, write_csv


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m sim.run", description="Simulate one warehouse scenario."
    )
    parser.add_argument("scenario", type=Path, help="scenario YAML file")
    parser.add_argument("--seed", type=int, help="override the seed of the scenario file")
    parser.add_argument("--out", type=Path, help="write the event log to this CSV file")
    parser.add_argument(
        "--plot-orders",
        nargs="?",
        const="show",
        metavar="FILE",
        help="plot the distribution of the generated orders (in a window, or saved to FILE)",
    )
    return parser.parse_args(argv)


def summary(sc: Scenario, result: Result, orders: list[Order]) -> str:
    """Human-readable report of one run."""
    s = result.state
    arrived = sum(isinstance(e.event, NewOrder) for e in result.log)
    if sc.mode == "continuous":
        mode = f"continuous (lambda = {sc.rate:g} /s, duration = {sc.duration:g} s)"
    else:
        mode = f"batch (N = {sc.orders}), finished: {'yes (s_fin)' if result.finished else 'NO'}"
    return "\n".join(
        [
            f"scenario:   {sc.name}  [{sc.layout.name}]",
            f"mode:       {mode}",
            f"seed:       {sc.seed}",
            f"orders:     {len(orders)} generated, {arrived} arrived, "
            f"{s.delivered} delivered, {len(s.queue)} in queue",
            f"sim time:   {result.time:.1f} s, {len(result.log)} events",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sc = load_scenario(args.scenario)
    if args.seed is not None:
        sc = dataclasses.replace(sc, seed=args.seed)
    result, orders = run_scenario(sc)

    print(summary(sc, result, orders))
    if args.out is not None:
        write_csv(result.log, args.out)
        print(f"log:        {args.out}")
    if args.plot_orders is not None:
        save = None if args.plot_orders == "show" else Path(args.plot_orders)
        rate = sc.rate if sc.mode == "continuous" else None
        title = f"{sc.name} (seed {sc.seed})"
        plot_orders(load_layout(sc.layout), orders, rate, title, save)
    # a batch run that does not reach s_fin is a failure (e.g. for scripts and CI)
    return 1 if sc.mode == "batch" and not result.finished else 0


if __name__ == "__main__":
    sys.exit(main())
