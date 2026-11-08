"""Throwaway demo: one batch simulation on the main map."""

import random
import time
from collections import Counter
from pathlib import Path

from fleet.guards import Model
from fleet.layout import load_layout
from fleet.routes import build_routes
from fleet.state import Order, initial_state
from fleet.supervisor import BaselinePolicy, Supervisor
from sim.simulator import Simulator, write_csv

LAYOUTS = Path(__file__).resolve().parents[1] / "layouts"

layout = load_layout(LAYOUTS / "warehouse_15x8.yaml")
model = Model(layout=layout, routes=build_routes(layout), q_max=20, b_res=3, capacity=400)

rng = random.Random(1)
orders = sorted(
    (
        Order(k, rng.choice(layout.pickups), rng.choice(layout.deliveries), rng.uniform(0, 600))
        for k in range(30)
    ),
    key=lambda o: o.created_at,
)

sim = Simulator(model, Supervisor(model, BaselinePolicy(b_low=100)), seed=1)
t0 = time.perf_counter()
result = sim.run(initial_state(layout), orders)
wall = time.perf_counter() - t0

print(f"robots: {len(result.state.robots)}, orders: {len(orders)}")
print(f"delivered: {result.state.delivered}, finished: {result.finished}")
print(f"warehouse time: {result.time:.1f} s, computed in {wall:.2f} s")
print(f"log entries: {len(result.log)}")
print("events:", dict(Counter(type(e.event).__name__ for e in result.log)))
print("\nfirst 12 log entries:")
for e in result.log[:12]:
    print(f"  {e.time:8.3f}  {type(e.event).__name__:9} robot={e.robot}  cell={e.cell}")

out = Path(__file__).with_name("demo_log.csv")
write_csv(result.log, out)
print(f"\nCSV: {out}")
