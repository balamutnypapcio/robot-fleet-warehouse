"""Deadlock-avoidance test safe(s, e) (proposal Sec. 2.6, Algorithm 1).

A state is safe if there is an order in which all active robots can drive to their reserved
goals one after another. Robot j is active if it holds a lane cell (H_j is not a subset of B).

    A <- active robots
    F <- all cells V without the cells held by robots in A
    repeat:
        find j in A such that rest_j is a subset of F
        if found: A <- A without j, F <- F with the cells of H_j added
    until no such j exists
    the state is safe iff A is empty

A robot that reaches its goal stands in a dead-end buffer cell that lies on no other route, so
it blocks nobody and all its cells can be released at once.
"""

from __future__ import annotations

from fleet.events import Event, Enter
from fleet.guards import Model
from fleet.state import SystemState, rest
from fleet.transitions import next_state


def is_safe_state(s: SystemState, model: Model) -> bool:
    """True iff all active robots of s can finish one after another (the loop above)."""
    active_robots = [
        i for i in range(len(s.robots)) if not set(s.robots[i].held) <= model.layout.buffer_cells
    ]
    occ_cells = set()
    for i in active_robots:
        occ_cells |= set(s.robots[i].held)
    empty_cells = model.layout.floor - occ_cells
    found = False
    while active_robots:
        for j in range(len(active_robots)):
            if rest(s, active_robots[j], model.routes) <= empty_cells:
                empty_cells |= set(s.robots[active_robots[j]].held)
                active_robots.pop(j)
                found = True
                break
        if not found:
            return False
        found = False
    return True


def safe(s: SystemState, e: Event, model: Model) -> bool:
    """safe(s, e): enter(i, v) is safe iff next_state(s, e) is a safe state; others always are."""
    if not isinstance(e,Enter):
        return True
    return is_safe_state(next_state(s,e,model),model)