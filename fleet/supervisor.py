"""Supervisor: which controllable events to issue in a state (proposal Sec. 2.7).

The Supervisor keeps the system safe and the Policy decides what is worth doing:

    candidates = {e in Econ and EF(s) : safe(s, e)}     computed by the Supervisor
    the next event, or None                             chosen by the Policy

After each chosen event the candidates are computed again from the new state. This resolves all
conflicts (one robot per cell, order and charger): a taken cell or order stops being enabled. It
also keeps a list of events safe as a whole, not only one by one.

Nothing here depends on time, graphics or ROS, so the simulator and the ROS 2 node can share it.
"""

from __future__ import annotations

from random import Random
from typing import Protocol

from fleet.events import (
    CONTROLLABLE,
    Assign,
    Charge,
    Dispatch,
    Enter,
    Event,
    Load,
    Unload,
)
from fleet.guards import Model, enabled_events
from fleet.safety import safe
from fleet.state import Mode, SystemState
from fleet.transitions import next_state


class Policy(Protocol):
    """Interface of a policy. The priority rules R1-R5 (issue #16) will be another Policy."""

    def choose(self, s: SystemState, candidates: list[Event], model: Model) -> Event | None:
        """Return one event from candidates, or None if nothing more should be issued now."""
        ...


class Supervisor:
    """Issues only events that are controllable, enabled and safe, whatever the policy does."""

    def __init__(self, model: Model, policy: Policy) -> None:
        """Remember the model and the policy."""
        self.model = model
        self.policy = policy

    def candidates(self, s: SystemState) -> list[Event]:
        """Controllable events of EF(s) for which safe(s, e) holds."""
        return [
            e
            for e in enabled_events(s, self.model)
            if (type(e) in CONTROLLABLE) and (safe(s, e, self.model))
        ]

    def decide(self, s: SystemState) -> list[Event]:
        """Events to issue now, to be applied in the returned order.

        Repeat: compute the candidates, ask the policy, apply the chosen event to a local copy of
        the state. Stop when the policy returns None. Always ask the policy at least once, even
        with no candidates, because a policy may need to see every state. Raise ValueError if the
        policy chooses something that is not a candidate.
        """
        event = []
        while True:
            candidates = self.candidates(s)
            e = self.policy.choose(s, candidates, self.model)
            if not e:
                return event
            if e not in candidates:
                raise ValueError("Broken policy, event is not in candidates")
            s = next_state(s, e, self.model)
            event.append(e)


class BaselinePolicy:
    """Baseline for the policy comparison (proposal Sec. 2.7, "Use for efficiency").

    The first rule that has a matching candidate decides:

    1. Start an operation: Load, Unload, Charge.
    2. Assign the oldest order in the queue to a random robot with battery >= b_low.
    3. Send a robot that has an order to its pick-up station, or to its delivery if it is Loaded.
    4. Send an idle robot without an order home:
         - battery < b_low and not on a charger: to a charger
         - on a pick-up or delivery station: to a parking place, or to a charger if none is free
         - any other idle robot stays where it is
    5. First-come-first-served cell access: of all Enter candidates, take the robot that has been
       standing longest on its current cell while wanting to move. Ties: lowest robot number.

    b_low must be at least the largest energy need of feas from a home cell. Otherwise a robot can
    be too weak for every order and still not count as low.
    """

    def __init__(self, b_low: int, seed: int = 0) -> None:
        """Remember b_low, create random.Random(seed) and prepare the memory for rule 5."""
        self.b_low = b_low
        self.random = Random(seed)
        self.moves_count = 0
        self.waiting = {}

    def choose(self, s: SystemState, candidates: list[Event], model: Model) -> Event | None:
        """Update the memory of waiting robots, then apply the rules above."""
        self.moves_count += 1
        for i, r in enumerate(s.robots):
            if r.goal is not None and len(r.held) == 1:
                if i not in self.waiting or self.waiting[i][0] != r.held[0]:
                    self.waiting[i] = (r.held[0], self.moves_count)
            else:
                self.waiting.pop(i, None)
        # rule 1
        for e in candidates:
            if type(e) in [Load, Unload, Charge]:
                return e
        # rule 2
        for o in s.queue:
            robot_idx = []
            for e in candidates:
                if type(e) is Assign and e.order == o:
                    if s.robots[e.robot].battery >= self.b_low:
                        robot_idx.append(e.robot)
            if robot_idx:
                return Assign(self.random.choice(robot_idx), o)
        # rule 3
        for e in candidates:
            if type(e) is Dispatch and s.robots[e.robot].order is not None:
                return e
        # rule 4
        robot_dispaches = {}
        for e in candidates:
            r = s.robots[e.robot]
            if type(e) is Dispatch and r.mode is Mode.IDLE and r.goal is None:
                robot_dispaches.setdefault(e.robot, []).append(e)
        for i, events in robot_dispaches.items():
            r = s.robots[i]
            chargers = [e for e in events if e.goal in model.layout.chargers]
            parkings = [e for e in events if e.goal in model.layout.parkings]
            if r.battery < self.b_low and r.held[0] not in model.layout.chargers and chargers:
                return chargers[0]
            if r.held[0] in (model.layout.pickups + model.layout.deliveries):
                if parkings:
                    return parkings[0]
                elif chargers:
                    return chargers[0]

        enters = [e for e in candidates if type(e) is Enter]
        if enters:
            return min(enters, key=lambda e: (self.waiting[e.robot][1], e.robot))

        return None
