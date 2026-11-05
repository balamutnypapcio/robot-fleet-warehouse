"""Events of the discrete-event model (proposal Sec. 2.3, Table 1)."""

from __future__ import annotations

from dataclasses import dataclass

from fleet.layout import Cell
from fleet.state import Order


@dataclass(frozen=True, slots=True)
class NewOrder:
    """order(o): a new order arrives from the warehouse management system."""

    order: Order


@dataclass(frozen=True, slots=True)
class Assign:
    """assign(i, o): order o is assigned to robot i."""

    robot: int
    order: Order


@dataclass(frozen=True, slots=True)
class Dispatch:
    """dispatch(i, g): robot i gets goal buffer cell g, which is reserved for it."""

    robot: int
    goal: Cell


@dataclass(frozen=True, slots=True)
class Enter:
    """enter(i, v): robot i may start moving into the neighbouring cell v."""

    robot: int
    cell: Cell


@dataclass(frozen=True, slots=True)
class Arrive:
    """arrive(i): robot i has completed the move and released the previous cell."""

    robot: int


@dataclass(frozen=True, slots=True)
class Load:
    """load(i): start of loading at the pick-up station."""

    robot: int


@dataclass(frozen=True, slots=True)
class Loaded:
    """loaded(i): end of loading."""

    robot: int


@dataclass(frozen=True, slots=True)
class Unload:
    """unload(i): start of unloading at the delivery station."""

    robot: int


@dataclass(frozen=True, slots=True)
class Unloaded:
    """unloaded(i): end of unloading."""

    robot: int


@dataclass(frozen=True, slots=True)
class Charge:
    """charge(i): start of charging."""

    robot: int


@dataclass(frozen=True, slots=True)
class Charged:
    """charged(i): battery full, charging finished."""

    robot: int


Event = (
    NewOrder
    | Assign
    | Dispatch
    | Enter
    | Arrive
    | Load
    | Loaded
    | Unload
    | Unloaded
    | Charge
    | Charged
)

CONTROLLABLE: frozenset[type] = frozenset({Assign, Dispatch, Enter, Load, Unload, Charge})
UNCONTROLLABLE: frozenset[type] = frozenset({NewOrder, Arrive, Loaded, Unloaded, Charged})
