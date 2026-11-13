"""Issue #11, "Done when": long random runs with 10 robots and safe() on, zero violations.

Run from the repository root:

    python -m experiments.monitor_runs                  # 10 seeds, 100 orders each
    python -m experiments.monitor_runs --seeds 3 --orders 30

Then a sanity check: the same supervisor with safe() switched off, started next to the 4-cell
loop of the mini layout, must end in a deadlock that the monitor reports.
"""

import argparse
import random
import time
from pathlib import Path

from fleet.guards import Model
from fleet.layout import load_layout
from fleet.routes import build_routes
from fleet.state import Mode, Order, RobotState, SystemState, initial_state
from fleet.supervisor import BaselinePolicy, Supervisor
from sim.simulator import Simulator
from verification.monitors import Monitor, MonitorViolation

LAYOUTS = Path(__file__).resolve().parents[1] / "layouts"


def make_model(name: str) -> Model:
    layout = load_layout(LAYOUTS / name)
    return Model(layout=layout, routes=build_routes(layout), q_max=20, b_res=3, capacity=400)


def random_orders(model: Model, count: int, seed: int, horizon: float) -> list[Order]:
    rng = random.Random(seed)
    layout = model.layout
    orders = [
        Order(k, rng.choice(layout.pickups), rng.choice(layout.deliveries), rng.uniform(0, horizon))
        for k in range(count)
    ]
    return sorted(orders, key=lambda o: o.created_at)


def long_runs(seeds: int, orders: int) -> None:
    model = make_model("warehouse_15x8.yaml")
    print(f"Long random runs: 10 robots, {orders} orders per run, safe() on")
    total_checked, total_violations = 0, 0
    for seed in range(seeds):
        monitor = Monitor(model)
        supervisor = Supervisor(model, BaselinePolicy(b_low=100, seed=seed))
        sim = Simulator(model, supervisor, seed=seed, monitor=monitor)
        start = time.perf_counter()
        try:
            result = sim.run(
                initial_state(model.layout),
                random_orders(model, orders, seed, horizon=orders * 30.0),
            )
            status = f"delivered {result.state.delivered}/{orders}, finished={result.finished}"
        except MonitorViolation as violation:
            total_violations += 1
            status = "VIOLATION\n" + str(violation)
        total_checked += monitor.checked
        wall = time.perf_counter() - start
        print(f"  seed {seed}: {monitor.checked} events checked, {status} ({wall:.1f} s)")
    print(f"Total: {total_checked} events checked, {total_violations} violations\n")


def sanity_check() -> None:
    model = make_model("mini.yaml")
    c2, p1, p2, d1 = (2, 0), (1, 3), (3, 3), (3, 0)
    s = SystemState(
        robots=(
            RobotState(((1, 1),), Mode.TO_DROP, 100, Order(3, p1, d1, 0.0), d1),
            RobotState(((2, 2),), Mode.TO_CHARGE, 100, None, c2),
            RobotState(((3, 1),), Mode.TO_PICK, 100, Order(2, p2, d1, 0.0), p2),
            RobotState(((3, 2),), Mode.TO_PICK, 100, Order(1, p1, d1, 0.0), p1),
        )
    )
    print("Sanity check on mini, safe() off: the monitor must report a deadlock")
    supervisor = Supervisor(model, BaselinePolicy(b_low=100), use_safe=False)
    try:
        Simulator(model, supervisor, monitor=Monitor(model)).run(s, [])
        print("  FAILED: no violation reported")
    except MonitorViolation as violation:
        print("  OK, reported:\n" + str(violation))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--orders", type=int, default=100)
    args = parser.parse_args()
    long_runs(args.seeds, args.orders)
    sanity_check()


if __name__ == "__main__":
    main()
