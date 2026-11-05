"""Enabling function EF(s): which events can occur in a state (proposal Sec. 2.4, Table 2).

    Event           Guard
    order(o)        |Q| < Qmax
    assign(i, o)    o in Q, m_i = Idle, o_i = none, |H_i| = 1, feas(i, o)
    dispatch(i, g)  |H_i| = 1, pos_i in B, free(g) and one of:
                      (a) m_i = Idle, o_i != none, g = pickup of o_i
                      (b) m_i = Loaded, g = delivery of o_i
                      (c) m_i = Idle, o_i = none, g in C
                      (d) m_i = Idle, o_i = none, g in K
    enter(i, v)     m_i in Mmov, g_i != none, |H_i| = 1, v = next_i, occ(v) = none, b_i >= 1
    arrive(i)       |H_i| = 2
    load(i)         m_i = ToPick, H_i = (pickup of o_i)
    loaded(i)       m_i = Loading
    unload(i)       m_i = ToDrop, H_i = (delivery of o_i)
    unloaded(i)     m_i = Unloading
    charge(i)       m_i = ToCharge, H_i = (c), c in C, g_i = none
    charged(i)      m_i = Charging

Mmov = {ToPick, ToDrop, ToCharge, ToPark}.

The condition g_i = none in charge(i) is not in Table 2. Without it a robot standing on one
charger and heading to another could start charging at once and keep its reservation forever.
"""

from __future__ import annotations

from dataclasses import dataclass

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
from fleet.layout import Layout
from fleet.routes import Routes
from fleet.state import Mode, Order, SystemState, free, next_cell, occ, pos


@dataclass(frozen=True, slots=True)
class Model:
    """Fixed data of the model, which never changes between states."""

    layout: Layout
    routes: Routes
    q_max: int  # Qmax: maximum length of the order queue
    b_res: int  # b_res: safety reserve of battery, in moves


def feas(s: SystemState, i: int, o: Order, model: Model) -> bool:
    """feas(i, o): enough energy for pick-up, delivery, the farthest charger and the reserve.

    b_i >= delta(pos_i, p_o) + delta(p_o, d_o) + max over c in C of delta(d_o, c) + b_res
    """
    pick_cell = o.pickup
    delivery_cell = o.delivery
    initial_robot_cell = pos(s, i)
    chargers_set = model.layout.chargers

    # find distance to farthest charging cell
    distance_from_delivery_to_charging = 0
    for c in chargers_set:
        distance_to_c = model.routes.delta(delivery_cell, c)
        if distance_to_c >= distance_from_delivery_to_charging:
            distance_from_delivery_to_charging = distance_to_c

    # compute distance
    distance = (
        model.routes.delta(initial_robot_cell, pick_cell)
        + model.routes.delta(pick_cell, delivery_cell)
        + distance_from_delivery_to_charging
    )
    return s.robots[i].battery >= distance + model.b_res


def is_enabled(s: SystemState, e: Event, model: Model) -> bool:
    """True iff the guard of event e holds in state s (see the table above)."""
    match e:
        case NewOrder(order=o):
            return len(s.queue) < model.q_max
        case Assign(robot=i, order=o):
            return (
                o in s.queue
                and s.robots[i].mode is Mode.IDLE
                and s.robots[i].order is None
                and len(s.robots[i].held) == 1
                and feas(s, i, o, model)
            )
        case Dispatch(robot=i, goal=g):
            on_buffer_cell = (
                len(s.robots[i].held) == 1
                and s.robots[i].held[0] in model.layout.buffer_cells
                and free(s, g)
            )
            idle_and_goal_pickup = (
                s.robots[i].mode is Mode.IDLE
                and s.robots[i].order is not None
                and g == s.robots[i].order.pickup
            )
            loaded_and_goal_delivery = (
                s.robots[i].mode is Mode.LOADED and g == s.robots[i].order.delivery
            )
            idle_no_order_and_goal_to_charger = (
                s.robots[i].mode is Mode.IDLE
                and s.robots[i].order is None
                and g in model.layout.chargers
            )
            idle_no_order_goal_to_parking = (
                s.robots[i].mode is Mode.IDLE
                and s.robots[i].order is None
                and g in model.layout.parkings
            )
            return on_buffer_cell and (
                idle_and_goal_pickup
                or loaded_and_goal_delivery
                or idle_no_order_and_goal_to_charger
                or idle_no_order_goal_to_parking
            )
        case Enter(robot=i, cell=v):
            r = s.robots[i]
            return (
                r.mode in (Mode.TO_PICK, Mode.TO_DROP, Mode.TO_CHARGE, Mode.TO_PARK)
                and r.goal is not None
                and len(r.held) == 1
                and v == next_cell(s, i, model.routes)
                and occ(s, v) is None
                and r.battery >= 1
            )
        case Arrive(robot=i):
            return len(s.robots[i].held) == 2
        case Load(robot=i):
            return s.robots[i].mode is Mode.TO_PICK and s.robots[i].held == (
                s.robots[i].order.pickup,
            )
        case Loaded(robot=i):
            return s.robots[i].mode is Mode.LOADING
        case Unload(robot=i):
            return s.robots[i].mode is Mode.TO_DROP and s.robots[i].held == (
                s.robots[i].order.delivery,
            )
        case Unloaded(robot=i):
            return s.robots[i].mode is Mode.UNLOADING
        case Charge(robot=i):
            return (
                s.robots[i].mode is Mode.TO_CHARGE
                and s.robots[i].held[0] in model.layout.chargers
                and len(s.robots[i].held) == 1
                and s.robots[i].goal is None
            )
        case Charged(robot=i):
            return s.robots[i].mode is Mode.CHARGING


def enabled_events(s: SystemState, model: Model) -> list[Event]:
    """EF(s): every enabled event, each listed once.

    NewOrder is not listed: its order comes from outside the state, so the caller that has
    a new order checks it with is_enabled.
    """
    events = (
        Assign,
        Dispatch,
        Enter,
        Arrive,
        Load,
        Loaded,
        Unload,
        Unloaded,
        Charge,
        Charged,
    )
    possible_events = []
    for e in events:
        if e in (Arrive, Load, Loaded, Unload, Unloaded, Charge, Charged):
            for i in range(len(s.robots)):
                if is_enabled(s, e(robot=i), model):
                    possible_events.append(e(robot=i))
        if e is Assign:
            for i in range(len(s.robots)):
                for o in s.queue:
                    if is_enabled(s, e(robot=i, order=o), model):
                        possible_events.append(e(robot=i, order=o))
        if e is Dispatch:
            for i in range(len(s.robots)):
                for c in model.layout.buffer_cells:
                    if is_enabled(s, e(robot=i, goal=c), model):
                        possible_events.append(e(robot=i, goal=c))
        if e is Enter:
            for i in range(len(s.robots)):
                v = next_cell(s, i, model.routes)
                if is_enabled(s, e(robot=i, cell=v), model):
                    possible_events.append(e(robot=i, cell=v))
    return possible_events
