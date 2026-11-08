"""Animation of one simulation run (a quick viewer; the full one is issue #13).

    python experiments/animate.py                                  # main map, 30 orders
    python experiments/animate.py --map mini --orders 8 --speed 4
    python experiments/animate.py --save run.gif                   # GIF instead of a window

Space pauses and resumes. Robots are coloured by mode, an x marks the goal of a robot in the
same colour, and lane cells show their exit directions.
"""

from __future__ import annotations

import argparse
import bisect
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle

from fleet.events import Arrive, Enter
from fleet.guards import Model
from fleet.layout import Cell, CellKind, Direction, Layout, load_layout
from fleet.routes import build_routes
from fleet.state import Mode, Order, SystemState, initial_state
from fleet.supervisor import BaselinePolicy, Supervisor
from fleet.transitions import next_state
from sim.simulator import Result, Simulator

LAYOUTS = Path(__file__).resolve().parents[1] / "layouts"
FPS = 25

CELL_COLOURS = {  # Fig. 1 of the proposal
    CellKind.LANE: "#ededed",
    CellKind.RACK: "#6e6e76",
    CellKind.PICKUP: "#78be78",
    CellKind.DELIVERY: "#f0a050",
    CellKind.CHARGER: "#6ea0e6",
    CellKind.PARKING: "#c8dcf5",
}
INTERSECTION_COLOUR = "#fae178"
MODE_COLOURS = {
    Mode.IDLE: "#9e9e9e",
    Mode.TO_PICK: "#2e9e44",
    Mode.LOADING: "#1b5e20",
    Mode.LOADED: "#e65100",
    Mode.TO_DROP: "#f57c00",
    Mode.UNLOADING: "#bf360c",
    Mode.TO_CHARGE: "#1e63c8",
    Mode.CHARGING: "#0d2f6e",
    Mode.TO_PARK: "#8e44ad",
}
ARROWS = {Direction.N: "↑", Direction.E: "→", Direction.S: "↓", Direction.W: "←"}


@dataclass(frozen=True, slots=True)
class Move:
    """One cell-to-cell move of a robot, from its Enter to its Arrive."""

    start: float
    end: float
    source: Cell
    target: Cell


class Timeline:
    """All states of a run, so that the state and the robot positions can be read at any time."""

    def __init__(self, layout: Layout, model: Model, result: Result) -> None:
        s = initial_state(layout)
        self.times = [0.0]
        self.states = [s]
        self.moves: list[list[Move]] = [[] for _ in s.robots]
        started: dict[int, tuple[float, Cell, Cell]] = {}
        for entry in result.log:
            e = entry.event
            if isinstance(e, Enter):
                started[e.robot] = (entry.time, s.robots[e.robot].held[0], e.cell)
            s = next_state(s, e, model)
            if isinstance(e, Arrive):
                start, source, target = started.pop(e.robot)
                self.moves[e.robot].append(Move(start, entry.time, source, target))
            self.times.append(entry.time)
            self.states.append(s)
        self.end = result.time
        self._move_starts = [[m.start for m in moves] for moves in self.moves]

    def state_at(self, t: float) -> SystemState:
        """The state after every event with time <= t."""
        return self.states[bisect.bisect_right(self.times, t) - 1]

    def position_at(self, i: int, t: float, s: SystemState) -> tuple[float, float]:
        """Centre of robot i, in between two cells while it moves."""
        k = bisect.bisect_right(self._move_starts[i], t) - 1
        if k >= 0:
            m = self.moves[i][k]
            if t < m.end:
                f = (t - m.start) / (m.end - m.start)
                return (
                    m.source[0] + f * (m.target[0] - m.source[0]),
                    m.source[1] + f * (m.target[1] - m.source[1]),
                )
        x, y = s.robots[i].held[0]
        return (float(x), float(y))


def simulate(
    layout: Layout, n_orders: int, seed: int, horizon: float
) -> tuple[Model, list[Order], Result]:
    """Batch run: n_orders with random stations, created uniformly in [0, horizon]."""
    model = Model(layout=layout, routes=build_routes(layout), q_max=20, b_res=3, capacity=400)
    rng = random.Random(seed)
    orders = sorted(
        (
            Order(
                k,
                rng.choice(layout.pickups),
                rng.choice(layout.deliveries),
                rng.uniform(0, horizon),
            )
            for k in range(n_orders)
        ),
        key=lambda o: o.created_at,
    )
    sim = Simulator(model, Supervisor(model, BaselinePolicy(b_low=100, seed=seed)), seed=seed)
    return model, orders, sim.run(initial_state(layout), orders)


def draw_layout(ax: plt.Axes, layout: Layout) -> None:
    for x in range(layout.width):
        for y in range(layout.height):
            cell = (x, y)
            kind = layout.kind(cell)
            if kind is CellKind.VOID:
                continue
            colour = INTERSECTION_COLOUR if cell in layout.intersections else CELL_COLOURS[kind]
            ax.add_patch(
                Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=colour, edgecolor="white", lw=1)
            )
            if kind is CellKind.LANE:
                arrows = "".join(ARROWS[d] for d in layout.exits[cell])
                ax.text(x, y, arrows, ha="center", va="center", fontsize=8, color="#a0a0a0")
            name = layout.buffer_name.get(cell)
            if name is not None:
                ax.text(x, y - 0.42, name, ha="center", va="bottom", fontsize=7, color="#222222")
    ax.set_xlim(-0.5, layout.width - 0.5)
    ax.set_ylim(-0.5, layout.height - 0.5)
    ax.set_aspect("equal")
    ax.axis("off")


def hud_text(t: float, s: SystemState, total: int, capacity: int) -> str:
    lines = [
        f"time       {t:9.1f} s",
        f"delivered  {s.delivered:4d} / {total}",
        f"queue      {len(s.queue):4d}",
        "",
        "robot  mode        battery",
    ]
    for i, r in enumerate(s.robots):
        lines.append(f"R{i:<5} {r.mode.name:<10}  {r.battery:4d}/{capacity}")
    return "\n".join(lines)


def build_animation(
    layout: Layout, model: Model, orders: list[Order], result: Result, speed: float
) -> tuple[plt.Figure, FuncAnimation]:
    timeline = Timeline(layout, model, result)
    n_robots = len(timeline.states[0].robots)

    fig = plt.figure(figsize=(14, 6.5))
    grid = fig.add_gridspec(1, 2, width_ratios=[max(layout.width, 6), 7])
    ax = fig.add_subplot(grid[0])
    side = fig.add_subplot(grid[1])
    side.axis("off")
    fig.suptitle(f"{layout.name}: {len(orders)} orders, {n_robots} robots")
    draw_layout(ax, layout)

    robots = []
    for i in range(n_robots):
        circle = Circle((0, 0), 0.33, zorder=3, edgecolor="black", lw=0.8)
        ax.add_patch(circle)
        label = ax.text(
            0,
            0,
            str(i),
            ha="center",
            va="center",
            fontsize=8,
            color="white",
            zorder=4,
            fontweight="bold",
        )
        robots.append((circle, label))
    goals = ax.scatter([], [], marker="x", s=60, linewidths=2, zorder=2)

    hud = side.text(
        0.02,
        0.98,
        "",
        va="top",
        ha="left",
        family="monospace",
        fontsize=9,
        transform=side.transAxes,
    )
    side.legend(
        handles=[
            Line2D([], [], marker="o", ls="", markersize=9, color=c, label=m.name)
            for m, c in MODE_COLOURS.items()
        ],
        loc="lower left",
        ncol=3,
        fontsize=8,
        frameon=False,
    )

    dt = speed / FPS
    frames = int(timeline.end / dt) + 2

    def update(k: int) -> None:
        t = min(k * dt, timeline.end)
        s = timeline.state_at(t)
        for i, (circle, label) in enumerate(robots):
            xy = timeline.position_at(i, t, s)
            circle.center = xy
            circle.set_facecolor(MODE_COLOURS[s.robots[i].mode])
            label.set_position(xy)
        with_goal = [r for r in s.robots if r.goal is not None]
        goals.set_offsets(np.array([r.goal for r in with_goal], dtype=float).reshape(-1, 2))
        goals.set_color([MODE_COLOURS[r.mode] for r in with_goal])
        hud.set_text(hud_text(t, s, len(orders), model.capacity))

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / FPS, repeat=False)

    paused = [False]

    def toggle(event) -> None:
        if event.key == " ":
            if paused[0]:
                anim.resume()
            else:
                anim.pause()
            paused[0] = not paused[0]

    fig.canvas.mpl_connect("key_press_event", toggle)
    return fig, anim


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--map", default="warehouse_15x8", help="layout name in layouts/")
    parser.add_argument("--orders", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--horizon", type=float, default=600.0, help="orders come in [0, h] s")
    parser.add_argument("--speed", type=float, default=10.0, help="warehouse seconds per second")
    parser.add_argument("--save", type=Path, help="write a GIF instead of opening a window")
    args = parser.parse_args()

    layout = load_layout(LAYOUTS / f"{args.map}.yaml")
    model, orders, result = simulate(layout, args.orders, args.seed, args.horizon)
    print(
        f"delivered {result.state.delivered}/{len(orders)}, finished={result.finished}, "
        f"warehouse time {result.time:.1f} s"
    )
    fig, anim = build_animation(layout, model, orders, result, args.speed)
    if args.save is not None:
        anim.save(args.save, writer=PillowWriter(fps=FPS))
        print(f"saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
