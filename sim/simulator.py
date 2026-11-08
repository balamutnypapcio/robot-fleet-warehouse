"""Discrete-event simulator: jumps from event to event instead of waiting in real time.

The supervisor decides at once (controllable events take no time); the world answers later with
the uncontrollable events that end each started action:

    Enter  -> Arrive    after a 1-cell move
    Load   -> Loaded    after loading
    Unload -> Unloaded  after unloading
    Charge -> Charged   after charging

Main loop of Simulator.run:

    put NewOrder(o) for every order into the event queue at time o.created_at
    react: let the supervisor decide and apply its events
    while the event queue is not empty:
        stop if the next event is after the time limit (continuous mode)
        pop the earliest event, apply it with next_state
        react: let the supervisor decide and apply its events
    every applied event goes to the log, and every started action schedules its completion

A new order that finds the queue full (|Q| = Qmax) waits outside and enters as soon as there is
room again, so no order is lost.
"""

from __future__ import annotations

import csv
import heapq
import itertools
from dataclasses import dataclass
from pathlib import Path
from random import Random

from fleet.events import (
    Arrive,
    Charge,
    Charged,
    Dispatch,
    Enter,
    Event,
    Load,
    Loaded,
    NewOrder,
    Unload,
    Unloaded,
)
from fleet.guards import Model, is_enabled
from fleet.layout import Cell
from fleet.state import Mode, Order, SystemState
from fleet.supervisor import Supervisor
from fleet.transitions import next_state


@dataclass(frozen=True, slots=True)
class Durations:
    """Ranges (low, high) in seconds; each duration is drawn uniformly from its range."""

    move: tuple[float, float] = (1.0, 1.5)  # one cell
    loading: tuple[float, float] = (10.0, 20.0)
    unloading: tuple[float, float] = (10.0, 20.0)
    charging: tuple[float, float] = (300.0, 600.0)  # always to a full battery


@dataclass(frozen=True, slots=True)
class LogEntry:
    """One applied event. In the CSV file the event is written as its name, e.g. "Enter"."""

    time: float
    event: Event
    robot: int | None  # None for NewOrder
    cell: Cell | None


@dataclass(frozen=True, slots=True)
class Result:
    state: SystemState  # the last state
    time: float  # continuous mode: the time limit; batch mode: the time of the last event
    log: tuple[LogEntry, ...]
    finished: bool  # True iff the last state is s_fin


def completion(e: Event, durations: Durations, rng: Random) -> tuple[Event, float] | None:
    """The uncontrollable event that ends the action started by e, and after how many seconds.

    None for events that start no action (Assign, Dispatch and every uncontrollable event).
    """
    match e:
        case Enter(robot=i):
            return (Arrive(robot=i), rng.uniform(*durations.move))
        case Load(robot=i):
            return (Loaded(robot=i), rng.uniform(*durations.loading))
        case Unload(robot=i):
            return (Unloaded(robot=i), rng.uniform(*durations.unloading))
        case Charge(robot=i):
            return (Charged(robot=i), rng.uniform(*durations.charging))
        case _:
            return None


def is_final(s: SystemState, total_orders: int, model: Model) -> bool:
    """s_fin (proposal Sec. 2.2): n = total_orders, Q empty, every robot Idle on a charger or a
    parking place. An idle robot has no goal, and with all orders delivered no robot has an order.
    """
    if s.delivered != total_orders:
        return False
    if s.queue:
        return False
    home = model.layout.parkings + model.layout.chargers
    for r in s.robots:
        if r.mode is not Mode.IDLE:  # Mode.IDLE = no goal and no order
            return False
        if r.held[0] not in home:
            return False
    return True


def log_entry(time: float, e: Event, s: SystemState) -> LogEntry:
    """Log entry for e, applied at the given time; s is the state after e.

    cell: Enter -> the cell entered, Dispatch -> the goal, NewOrder -> the pick-up of the order,
    any other event -> the position of the robot after the event.
    """
    robot = None if isinstance(e, NewOrder) else e.robot
    match e:
        case Enter(cell=c):
            return LogEntry(time=time, event=e, robot=robot, cell=c)
        case Dispatch(goal=g):
            return LogEntry(time=time, event=e, robot=robot, cell=g)
        case NewOrder(order=o):
            return LogEntry(time=time, event=e, robot=robot, cell=o.pickup)
        case _:
            return LogEntry(time=time, event=e, robot=robot, cell=s.robots[e.robot].held[0])


def write_csv(log: tuple[LogEntry, ...], path: Path) -> None:
    """CSV with the header time,event,robot,cell. Example row: 12.500,Enter,3,4 6

    time with 3 decimals, event as its name, robot empty for NewOrder, cell as "x y".
    """
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "event", "robot", "cell"])
        for entry in log:
            robot = "" if entry.robot is None else entry.robot
            cell = "" if entry.cell is None else f"{entry.cell[0]} {entry.cell[1]}"
            writer.writerow([f"{entry.time:.3f}", type(entry.event).__name__, robot, cell])


class Simulator:
    def __init__(
        self,
        model: Model,
        supervisor: Supervisor,
        durations: Durations | None = None,
        seed: int = 0,
    ) -> None:
        """Remember the arguments (Durations() if durations is None), create Random(seed)."""
        self.model = model
        self.supervisor = supervisor
        self.durations = Durations() if durations is None else durations
        self._rng = Random(seed)

    def run(self, s: SystemState, orders: list[Order], time_limit: float | None = None) -> Result:
        """Simulate from state s (see the loop above).

        Stop when nothing more can happen (the event queue is empty) and, if time_limit is given
        (continuous mode), before the first event after time_limit. In batch mode the queue runs
        empty exactly when s_fin is reached, or when the system is stuck: finished tells which.

        heapq compares whole entries, and two events cannot be compared, so push entries
        (time, number, event) with an increasing number: it also keeps events with equal times in
        the order in which they were pushed.
        """
        heap = []
        log = []
        waiting = []
        counter = itertools.count()
        time = 0.0
        for o in orders:
            heapq.heappush(heap, ((o.created_at), next(counter), NewOrder(o)))

        def apply(s: SystemState, e: Event, now: float) -> SystemState:
            s = next_state(s, e, self.model)
            log.append(log_entry(now, e, s))
            c = completion(e, self.durations, self._rng)
            if c is not None:
                heapq.heappush(heap, (now + c[1], next(counter), c[0]))
            return s

        def react(s: SystemState, now: float) -> SystemState:
            while waiting and is_enabled(s, NewOrder(waiting[0]), self.model):
                s = apply(s, NewOrder(waiting[0]), now)
                waiting.pop(0)
            for e in self.supervisor.decide(s):
                s = apply(s, e, now)
            return s

        s = react(s, 0.0)
        while heap:
            t, _, e = heap[0]
            if time_limit is not None and t > time_limit:
                break
            time, _, e = heapq.heappop(heap)
            if isinstance(e, NewOrder):
                if not is_enabled(s, e, self.model):
                    waiting.append(e.order)
                    continue
            s = apply(s, e, time)
            s = react(s, time)

        return Result(
            s,
            time if time_limit is None else time_limit,
            tuple(log),
            is_final(s, len(orders), self.model),
        )
