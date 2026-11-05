"""Transition function next_state(s, e) (proposal Sec. 2.5, column "Effect" of Table 2).

Event           Effect (variables not mentioned remain unchanged)
order(o)        Q <- Q + o (append at the end)
assign(i, o)    o_i <- o, Q <- Q without o
dispatch(i, g)  g_i <- g, m_i <- one of:
                  (a) Idle with an order -> ToPick
                  (b) Loaded -> ToDrop
                  (c) Idle without an order, g in C -> ToCharge
                  (d) Idle without an order, g in K -> ToPark
enter(i, v)     H_i <- (pos_i, v)
arrive(i)       H_i = (u, v): H_i <- (v), b_i <- b_i - 1
                if v = g_i: g_i <- none and, if m_i = ToPark, m_i <- Idle
load(i)         m_i <- Loading
loaded(i)       m_i <- Loaded
unload(i)       m_i <- Unloading
unloaded(i)     m_i <- Idle, o_i <- none, n <- n + 1
charge(i)       m_i <- Charging
charged(i)      m_i <- Idle, b_i <- B
"""

from __future__ import annotations

from typing import Any

from dataclasses import replace

from fleet.events import (
    Arrive,
    Assign,
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
from fleet.state import SystemState, Mode
from fleet.layout import CellKind

_MODE_FOR_GOAL = {
    CellKind.PICKUP: Mode.TO_PICK,
    CellKind.DELIVERY: Mode.TO_DROP,
    CellKind.CHARGER: Mode.TO_CHARGE,
    CellKind.PARKING: Mode.TO_PARK,
}


def _with_robot(s: SystemState, i: int, **changes: Any) -> SystemState:
    """Copy of s in which the given fields of robot i are changed."""
    robots = list(s.robots)
    robots[i] = replace(robots[i], **changes)
    return replace(s, robots=tuple(robots))


def next_state(s: SystemState, e: Event, model: Model) -> SystemState:
    """s' = next_state(s, e); raise ValueError if e is not in EF(s)."""
    if not is_enabled(s, e, model):
        raise ValueError(f"event {e} is not in EF(s)")
    match e:
        case NewOrder(order=o):
            return replace(s, queue=s.queue + (o,))
        case Assign(robot=i, order=o):
            state = _with_robot(s, i, order=o)
            queue = tuple(x for x in s.queue if x != o)
            return replace(state, queue=queue)
        case Dispatch(robot=i, goal=g):
            mode = _MODE_FOR_GOAL[model.layout.kind(g)]
            return _with_robot(s, i, mode=mode, goal=g)
        case Enter(robot=i, cell=c):
            return _with_robot(s, i, held=(s.robots[i].held[0], c))
        case Arrive(robot=i):
            r = s.robots[i]
            held = r.held[1]
            goal = r.goal
            mode = r.mode
            if held == r.goal:
                goal = None
                if mode is Mode.TO_PARK:
                    mode = Mode.IDLE
            return _with_robot(s, i, held=(held,), mode=mode, goal=goal, battery=r.battery - 1)
        case Load(robot=i):
            return _with_robot(s, i, mode=Mode.LOADING)
        case Loaded(robot=i):
            return _with_robot(s, i, mode=Mode.LOADED)
        case Unload(robot=i):
            return _with_robot(s, i, mode=Mode.UNLOADING)
        case Unloaded(robot=i):
            state = replace(s, delivered=s.delivered+1)
            return _with_robot(state, i, mode=Mode.IDLE, order=None)
        case Charge(robot=i):
            return _with_robot(s, i, mode=Mode.CHARGING)
        case Charged(robot=i):
            return _with_robot(s, i, mode=Mode.IDLE, battery=model.capacity)