"""Scenario file (YAML): everything needed for one simulation run.

    name: default
    layout: ../layouts/warehouse_15x8.yaml   # relative to the scenario file
    seed: 1
    mode: continuous                         # continuous | batch
    continuous: {rate: 0.02, duration: 3600}  # lambda (orders per second), run length
    batch: {orders: 30}                       # N orders at t = 0
    queue_max: 20                            # Qmax
    battery: {capacity: 400, reserve: 3}     # B, b_res (in moves)
    policy: {b_low: 100}                     # parameter of the baseline policy
    durations: {move: [1.0, 1.5], ...}        # optional, defaults of sim.simulator.Durations

Only the block of the chosen mode is used; the other one may stay in the file, so switching
modes means changing one line.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import Random

import yaml

from fleet.guards import Model
from fleet.layout import load_layout
from fleet.routes import build_routes
from fleet.state import Order, initial_state
from fleet.supervisor import BaselinePolicy, Supervisor
from sim.orders import batch_orders, poisson_orders
from sim.simulator import Durations, Result, Simulator


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    layout: Path  # absolute path of the layout YAML
    seed: int
    mode: str  # "continuous" or "batch"
    rate: float | None  # lambda, continuous mode only
    duration: float | None  # time limit, continuous mode only
    orders: int | None  # N, batch mode only
    queue_max: int  # Qmax
    capacity: int  # B
    reserve: int  # b_res
    b_low: int
    durations: Durations


def load_scenario(path: str | Path) -> Scenario:
    """Read a scenario file; a relative layout path is taken relative to the file."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return parse_scenario(data, base_dir=path.resolve().parent)


def parse_scenario(data: dict, base_dir: Path) -> Scenario:
    """Build a Scenario from the parsed YAML mapping."""
    mode = data["mode"]
    if mode not in ("continuous", "batch"):
        raise ValueError(f"mode must be continuous or batch, got {mode!r}")
    continuous = mode == "continuous"
    ranges = data.get("durations", {})
    return Scenario(
        name=data.get("name", ""),
        layout=(base_dir / data["layout"]).resolve(),  # an absolute path stays as it is
        seed=data.get("seed", 0),
        mode=mode,
        rate=data["continuous"]["rate"] if continuous else None,
        duration=float(data["continuous"]["duration"]) if continuous else None,
        orders=None if continuous else data["batch"]["orders"],
        queue_max=data["queue_max"],
        capacity=data["battery"]["capacity"],
        reserve=data["battery"]["reserve"],
        b_low=data["policy"]["b_low"],
        durations=Durations(**{key: tuple(value) for key, value in ranges.items()}),
    )


def build_model(sc: Scenario) -> Model:
    """The fixed model data of the scenario: layout, routes, Qmax, b_res and B."""
    layout = load_layout(sc.layout)
    return Model(
        layout=layout,
        routes=build_routes(layout),
        q_max=sc.queue_max,
        b_res=sc.reserve,
        capacity=sc.capacity,
    )


def max_order_need(model: Model) -> int:
    """Largest energy feas can ask for from a home cell (a charger or a parking place):

    max over home h, pick-up p, delivery d of
        delta(h, p) + delta(p, d) + max over c in C of delta(d, c) + b_res
    """
    layout, delta = model.layout, model.routes.delta
    return max(
        delta(h, p) + delta(p, d) + max(delta(d, c) for c in layout.chargers) + model.b_res
        for h in layout.chargers + layout.parkings
        for p in layout.pickups
        for d in layout.deliveries
    )


def check_energy(sc: Scenario, model: Model) -> None:
    """Raise ValueError if a full battery or b_low is below the largest need of an order.

    Without this check a wrong scenario does not crash, the robots just get stuck:
    capacity too small: some order is never feasible, even for a fully charged robot;
    b_low too small: a robot can be too weak for every order and still not go charging
    (see BaselinePolicy).
    """
    need = max_order_need(model)
    if sc.capacity < need or sc.b_low < need:
        raise ValueError(
            f"battery.capacity ({sc.capacity}) and policy.b_low ({sc.b_low}) must be at least "
            f"{need}, the largest energy need of an order"
        )


def run_scenario(sc: Scenario) -> tuple[Result, list[Order]]:
    """Simulate the scenario from s0; return the result and the generated orders.

    The one scenario seed gives three separate random streams: orders (seed), durations
    (seed + 1) and the policy (seed + 2). Continuous mode stops at the duration; batch mode
    runs until nothing more can happen (result.finished tells whether s_fin was reached).
    Batch orders enter as NewOrder events at t = 0, so N > Qmax works: the simulator keeps the
    extra orders waiting until the queue has room.
    """
    model = build_model(sc)
    check_energy(sc, model)
    rng = Random(sc.seed)
    if sc.mode == "continuous":
        orders = poisson_orders(model.layout, sc.rate, sc.duration, rng)
    else:
        orders = batch_orders(model.layout, sc.orders, rng)
    policy = BaselinePolicy(b_low=sc.b_low, seed=sc.seed + 2)
    sim = Simulator(model, Supervisor(model, policy), durations=sc.durations, seed=sc.seed + 1)
    s0 = initial_state(model.layout, capacity=sc.capacity)
    time_limit = sc.duration if sc.mode == "continuous" else None
    return sim.run(s0, orders, time_limit=time_limit), orders
