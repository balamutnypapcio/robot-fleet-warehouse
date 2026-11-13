"""Runtime monitors for the safety properties S1-S3 (proposal Sec. 2.6), issue #11.

The monitors are independent checks of the result: they look only at the state and never call
safe() or is_safe_state(), so a bug in safe() cannot hide a violation.

    S1  collision freedom: no cell is held by two robots
    S2  energy safety:     battery_i >= delta(pos_i, nearest charger) for every robot
    S3  deadlock freedom:  no cycle in the wait-for graph

Wait-for graph: robot i waits for robot j if i has a goal, stands still in one cell (|H_i| = 1)
and the next cell of its route is held by j. Each robot waits for at most one other robot, so a
cycle is found by following the arrows from every robot.

Usage in the simulator: Monitor.check(time, state, log) after every applied event. On a violation
it raises MonitorViolation, which stops the simulation and carries a readable report.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from fleet.guards import Model
from fleet.layout import Cell
from fleet.state import SystemState


@dataclass(frozen=True, slots=True)
class Violation:
    prop: str  # "S1", "S2" or "S3"
    robots: tuple[int, ...]  # robots involved, numbered from 0
    message: str


def holders(s: SystemState) -> dict[Cell, list[int]]:
    """For every held cell, the list of robots holding it."""
    result: dict[Cell, list[int]] = {}
    for i, r in enumerate(s.robots):
        for c in r.held:
            result.setdefault(c, []).append(i)
    return result


def check_collision(s: SystemState) -> list[Violation]:
    """S1: a violation for every cell held by more than one robot."""
    violations = []
    for cell, robots in holders(s).items():
        if len(robots) > 1:
            violations.append(Violation("S1", tuple(robots), f"robots {robots} share cell {cell}"))
    return violations


def distance_to_nearest_charger(model: Model, u: Cell) -> int:
    """delta(u, C), computed here from the routes (not taken from the guards)."""
    return min(model.routes.delta(u, c) for c in model.layout.chargers)


def check_energy(s: SystemState, model: Model) -> list[Violation]:
    """S2: a violation for every robot that cannot reach a charger with its battery."""
    violations = []
    for i, r in enumerate(s.robots):
        u = r.held[0]
        need = distance_to_nearest_charger(model, u)
        if r.battery < need:
            violations.append(
                Violation(
                    "S2",
                    (i,),
                    f"robot {i} at {u} has battery {r.battery} < {need} moves to a charger",
                )
            )
    return violations


def wait_for_graph(s: SystemState, model: Model) -> dict[int, int]:
    """Arrows i -> j: robot i stands still and the next cell of its route is held by j."""
    owner = {}
    for i, r in enumerate(s.robots):
        for c in r.held:
            owner[c] = i
    graph = {}
    for i, r in enumerate(s.robots):
        if r.goal is None or len(r.held) != 1 or r.held[0] == r.goal:
            continue  # no goal, already moving, or already there: waits for nobody
        v = model.routes.next_cell(r.held[0], r.goal)
        j = owner.get(v)
        if j is not None and j != i:
            graph[i] = j
    return graph


def find_cycles(graph: dict[int, int]) -> list[tuple[int, ...]]:
    """All cycles of a graph in which every node has at most one outgoing arrow."""
    cycles = []
    seen_cycles = set()
    for start in graph:
        path = [start]
        node = graph[start]
        while node in graph and node not in path:
            path.append(node)
            node = graph[node]
        if node in path:  # we came back to a robot of this walk: cycle
            cycle = tuple(path[path.index(node) :])
            if frozenset(cycle) not in seen_cycles:
                seen_cycles.add(frozenset(cycle))
                cycles.append(cycle)
    return cycles


def find_deadlock(s: SystemState, model: Model) -> list[Violation]:
    """S3: a violation for every circular wait."""
    violations = []
    for cycle in find_cycles(wait_for_graph(s, model)):
        arrows = " -> ".join(str(i) for i in cycle + (cycle[0],))
        violations.append(Violation("S3", cycle, f"circular wait {arrows}"))
    return violations


def check_all(s: SystemState, model: Model) -> list[Violation]:
    """S1, S2 and S3 together."""
    return check_collision(s) + check_energy(s, model) + find_deadlock(s, model)


def format_report(
    time: float, violations: list[Violation], s: SystemState, last_events: Sequence
) -> str:
    """Readable report: time, violations, state of every robot and the last log entries."""
    lines = [f"Safety violation at t = {time:.3f} s"]
    for v in violations:
        lines.append(f"  {v.prop}: {v.message}")
    lines.append(f"State: queue = {len(s.queue)} orders, delivered = {s.delivered}")
    for i, r in enumerate(s.robots):
        lines.append(
            f"  robot {i}: held={r.held} mode={r.mode.name} battery={r.battery} goal={r.goal}"
        )
    lines.append(f"Last {len(last_events)} events:")
    for e in last_events:
        lines.append(f"  {e.time:9.3f}  {type(e.event).__name__:9} robot={e.robot}  cell={e.cell}")
    return "\n".join(lines)


class MonitorViolation(Exception):
    """Raised by Monitor.check; stops the simulation. str(exc) is the full report."""

    def __init__(
        self, time: float, violations: list[Violation], state: SystemState, last_events: Sequence
    ) -> None:
        super().__init__(format_report(time, violations, state, last_events))
        self.time = time
        self.violations = violations
        self.state = state
        self.last_events = tuple(last_events)


class Monitor:
    """Checks S1-S3 after every event; history = how many last log entries a report shows."""

    def __init__(self, model: Model, history: int = 10) -> None:
        self.model = model
        self.history = history
        self.checked = 0  # number of checked states

    def check(self, time: float, s: SystemState, log: Sequence = ()) -> None:
        """Raise MonitorViolation if s violates S1, S2 or S3."""
        self.checked += 1
        violations = check_all(s, self.model)
        if violations:
            raise MonitorViolation(time, violations, s, log[-self.history :])
