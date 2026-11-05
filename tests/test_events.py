import dataclasses

import pytest

from fleet import events
from fleet.events import (
    Arrive,
    Assign,
    Charge,
    Charged,
    Dispatch,
    Enter,
    Load,
    Loaded,
    NewOrder,
    Unload,
    Unloaded,
)
from fleet.state import Order

ORDER = Order(id=1, pickup=(1, 3), delivery=(3, 0), created_at=0.0)

ALL_EVENTS = [
    NewOrder(ORDER),
    Assign(0, ORDER),
    Dispatch(0, (1, 3)),
    Enter(0, (1, 2)),
    Arrive(0),
    Load(0),
    Loaded(0),
    Unload(0),
    Unloaded(0),
    Charge(0),
    Charged(0),
]


def test_controllable_events():
    assert events.CONTROLLABLE == {Assign, Dispatch, Enter, Load, Unload, Charge}


def test_uncontrollable_events():
    assert events.UNCONTROLLABLE == {NewOrder, Arrive, Loaded, Unloaded, Charged}


def test_every_event_type_is_in_exactly_one_set():
    assert events.CONTROLLABLE.isdisjoint(events.UNCONTROLLABLE)
    assert events.CONTROLLABLE | events.UNCONTROLLABLE == {type(e) for e in ALL_EVENTS}


def test_sets_are_immutable():
    assert isinstance(events.CONTROLLABLE, frozenset)
    assert isinstance(events.UNCONTROLLABLE, frozenset)


def test_events_are_hashable():
    assert len(set(ALL_EVENTS)) == len(ALL_EVENTS)


def test_events_compare_by_type_and_parameters():
    assert Enter(0, (1, 2)) == Enter(0, (1, 2))
    assert Enter(0, (1, 2)) != Enter(1, (1, 2))
    assert Enter(0, (1, 2)) != Enter(0, (2, 2))
    assert Load(0) != Loaded(0)


@pytest.mark.parametrize("event", ALL_EVENTS, ids=lambda e: type(e).__name__)
def test_events_are_immutable(event):
    field = dataclasses.fields(event)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(event, field, None)
