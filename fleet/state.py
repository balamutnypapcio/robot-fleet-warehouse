"""System state s = (s_1..s_n, Q, n) and the initial state s0 (proposal Sec. 2.2).

States are immutable and hashable, so copies are cheap and visited states can be kept in a set.
Robots are numbered from 0: robot 1 of the proposal is `robots[0]`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum, auto

from fleet.layout import Cell, Layout
from fleet.routes import Routes

BATTERY_CAPACITY = 400  # B: full battery, measured in moves


class Mode(Enum):
    IDLE = auto()
    TO_PICK = auto()
    LOADING = auto()
    LOADED = auto()
    TO_DROP = auto()
    UNLOADING = auto()
    TO_CHARGE = auto()
    CHARGING = auto()
    TO_PARK = auto()


@dataclass(frozen=True, slots=True)
class Order:
    id: int
    pickup: Cell
    delivery: Cell
    created_at: float  # given by the simulator; fleet/ itself never reads a clock


@dataclass(frozen=True, slots=True)
class RobotState:
    held: tuple[Cell, ...]  # H: (u) standing in u, (u, v) moving from u to v
    mode: Mode
    battery: int
    order: Order | None = None
    goal: Cell | None = None  # reserved buffer cell


@dataclass(frozen=True, slots=True)
class SystemState:
    robots: tuple[RobotState, ...]
    queue: tuple[Order, ...] = ()  # Q: FIFO of unassigned orders
    delivered: int = 0  # n


def pos(s: SystemState, i: int) -> Cell:
    """pos_i: the first cell held by robot i."""
    return s.robots[i].held[0]


def occ(s: SystemState, v: Cell) -> int | None:
    """occ(v): the robot holding cell v, or None if nobody holds it."""
    i = 0
    for r in s.robots:
        for h in r.held:
            if h == v:
                return i
        i += 1
    return None


def free(s: SystemState, g: Cell) -> bool:
    """free(g): nobody holds buffer cell g and nobody has reserved it as a goal."""
    not_reserved_as_goal = True
    for r in s.robots:
        if r.goal == g:
            not_reserved_as_goal = False
            break
    return occ(s, g) is None and not_reserved_as_goal


def next_cell(s: SystemState, i: int, routes: Routes) -> Cell | None:
    """next_i: second cell of route(pos_i, g_i), or None if robot i has no goal."""
    g = s.robots[i].goal
    if g is None:
        return None
    r = routes.route(pos(s, i), g)
    return r[1]


def rest(s: SystemState, i: int, routes: Routes) -> frozenset[Cell]:
    """rest_i: cells of route(pos_i, g_i) not yet held by robot i; empty if it has no goal."""
    g = s.robots[i].goal
    if g is None:
        return frozenset()
    whole_route = routes.route(pos(s, i), g)
    return frozenset(whole_route) - frozenset(s.robots[i].held)


def initial_state(
    layout: Layout, batch: Iterable[Order] | None = None, capacity: int = BATTERY_CAPACITY
) -> SystemState:
    """s0: first robots on chargers C1.., the rest on parking K1..; all Idle with a full battery.

    `batch` is the initial queue Q0 (batch scenario); None means an empty queue.
    """
    start_cells = layout.chargers + layout.parkings
    r = []
    for sc in start_cells:
        r.append(RobotState(held=(sc,), mode=Mode.IDLE, battery=capacity))
    q = () if batch is None else tuple(batch)
    return SystemState(robots=tuple(r), queue=q)
