import copy
import dataclasses
from pathlib import Path

import pytest

from fleet.layout import load_layout
from fleet.routes import build_routes
from fleet.state import (
    BATTERY_CAPACITY,
    Mode,
    Order,
    RobotState,
    SystemState,
    free,
    initial_state,
    next_cell,
    occ,
    pos,
    rest,
)

LAYOUTS = Path(__file__).resolve().parents[1] / "layouts"

#   .   P1  #   P2
#   K2  v   +<v <
#   K1  >   +>  ^
#   .   C1  C2  D1
K1, K2 = (0, 1), (0, 2)
C1, C2 = (1, 0), (2, 0)
P1, P2 = (1, 3), (3, 3)
D1 = (3, 0)
ORDER = Order(id=1, pickup=P1, delivery=D1, created_at=0.0)


def robot(*held, goal=None, mode=Mode.IDLE, battery=100, order=None):
    return RobotState(held=held, mode=mode, battery=battery, order=order, goal=goal)


def system(*robots, queue=(), delivered=0):
    return SystemState(robots=robots, queue=queue, delivered=delivered)


@pytest.fixture(scope="module")
def main():
    return load_layout(LAYOUTS / "warehouse_15x8.yaml")


@pytest.fixture(scope="module")
def mini():
    return load_layout(LAYOUTS / "mini.yaml")


@pytest.fixture(scope="module")
def mini_routes(mini):
    return build_routes(mini)


# data structures


def test_mode_has_all_operating_modes():
    assert {m.name for m in Mode} == {
        "IDLE",
        "TO_PICK",
        "LOADING",
        "LOADED",
        "TO_DROP",
        "UNLOADING",
        "TO_CHARGE",
        "CHARGING",
        "TO_PARK",
    }


def test_robot_state_defaults_to_no_order_and_no_goal():
    r = RobotState(held=(K1,), mode=Mode.IDLE, battery=10)
    assert r.order is None
    assert r.goal is None


def test_system_state_defaults_to_empty_queue_and_nothing_delivered():
    s = SystemState(robots=(robot(K1),))
    assert s.queue == ()
    assert s.delivered == 0


def test_states_are_immutable():
    r = robot(K1)
    s = system(r)
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.battery = 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.delivered = 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        ORDER.id = 2


def test_states_are_hashable():
    s = system(robot(K1, order=ORDER, goal=P1), robot(C1, (1, 1)), queue=(ORDER,))
    assert isinstance(hash(s), int)


def test_equal_states_count_as_one_visited_state():
    a = system(robot(K1, goal=P1), queue=(ORDER,))
    b = system(robot(K1, goal=P1), queue=(ORDER,))
    assert a == b
    assert len({a, b}) == 1


@pytest.mark.parametrize(
    "other",
    [
        system(robot(K2)),
        system(robot(K1, (1, 1))),
        system(robot(K1, mode=Mode.TO_PARK)),
        system(robot(K1, battery=99)),
        system(robot(K1, order=ORDER)),
        system(robot(K1, goal=P1)),
        system(robot(K1), queue=(ORDER,)),
        system(robot(K1), delivered=1),
    ],
)
def test_any_changed_field_gives_a_different_state(other):
    assert system(robot(K1)) != other
    assert len({system(robot(K1)), other}) == 2


def test_replace_leaves_the_old_state_unchanged():
    old = system(robot(K1))
    new = dataclasses.replace(old, delivered=1)
    assert old.delivered == 0
    assert new.delivered == 1
    assert new.robots is old.robots


# pos


def test_pos_standing_robot():
    assert pos(system(robot(K1), robot(C1)), 1) == C1


def test_pos_moving_robot_is_the_first_held_cell():
    assert pos(system(robot(K1, (1, 1))), 0) == K1


# occ


def test_occ_free_cell_is_none():
    assert occ(system(robot(K1)), C1) is None


def test_occ_returns_robot_index():
    s = system(robot(K1), robot(C1), robot(C2))
    assert occ(s, K1) == 0  # 0 is a valid robot, not "nobody"
    assert occ(s, C1) == 1
    assert occ(s, C2) == 2


def test_occ_moving_robot_holds_both_cells():
    s = system(robot(K2), robot(K1, (1, 1)))
    assert occ(s, K1) == 1
    assert occ(s, (1, 1)) == 1


# free


def test_free_when_nobody_holds_or_reserves_it():
    assert free(system(robot(K1, goal=P1)), P2) is True


def test_not_free_when_a_robot_stands_there():
    assert free(system(robot(K1), robot(P2)), P2) is False


def test_not_free_when_robot_is_driving_in():
    assert free(system(robot((3, 2), P2, goal=P2)), P2) is False


def test_not_free_when_robot_is_driving_out():
    assert free(system(robot(P2, (3, 2))), P2) is False


def test_not_free_when_reserved_by_a_distant_robot():
    assert free(system(robot(K1), robot(C1, goal=P2)), P2) is False


# next_cell


def test_next_cell_without_goal_is_none(mini_routes):
    assert next_cell(system(robot(K1)), 0, mini_routes) is None


def test_next_cell_standing_robot(mini_routes):
    assert next_cell(system(robot(K2, goal=P2)), 0, mini_routes) == (1, 2)


def test_next_cell_depends_on_the_goal(mini_routes):
    # The intersection (2, 2) has two exits.
    assert next_cell(system(robot((2, 2), goal=P2)), 0, mini_routes) == (2, 1)
    assert next_cell(system(robot((2, 2), goal=P1)), 0, mini_routes) == (1, 2)


def test_next_cell_uses_the_right_robot(mini_routes):
    s = system(robot(K1), robot((3, 2), goal=P2))
    assert next_cell(s, 0, mini_routes) is None
    assert next_cell(s, 1, mini_routes) == P2


def test_next_cell_moving_robot_is_the_cell_it_moves_into(mini_routes):
    assert next_cell(system(robot(K2, (1, 2), goal=P2)), 0, mini_routes) == (1, 2)


# rest


def test_rest_without_goal_is_empty(mini_routes):
    assert rest(system(robot(K1)), 0, mini_routes) == frozenset()


def test_rest_standing_robot_is_route_without_its_cell(mini_routes):
    assert rest(system(robot(K2, goal=P2)), 0, mini_routes) == {
        (1, 2),
        (1, 1),
        (2, 1),
        (3, 1),
        (3, 2),
        (3, 3),
    }


def test_rest_moving_robot_excludes_both_held_cells(mini_routes):
    assert rest(system(robot(K2, (1, 2), goal=P2)), 0, mini_routes) == {
        (1, 1),
        (2, 1),
        (3, 1),
        (3, 2),
        (3, 3),
    }


def test_rest_one_move_before_goal_is_only_the_goal(mini_routes):
    assert rest(system(robot((3, 2), goal=P2)), 0, mini_routes) == {P2}


def test_rest_moving_into_goal_is_empty(mini_routes):
    assert rest(system(robot((3, 2), P2, goal=P2)), 0, mini_routes) == frozenset()


def test_rest_is_a_frozenset(mini_routes):
    assert isinstance(rest(system(robot(K2, goal=P2)), 0, mini_routes), frozenset)
    assert isinstance(rest(system(robot(K1)), 0, mini_routes), frozenset)


def test_rest_uses_the_right_robot(mini_routes):
    s = system(robot(K1), robot((3, 2), goal=P2))
    assert rest(s, 0, mini_routes) == frozenset()
    assert rest(s, 1, mini_routes) == {P2}


# initial_state


def test_initial_state_main_robot_cells(main):
    s0 = initial_state(main)
    assert [r.held for r in s0.robots] == [
        (main.buffers[name],)
        for name in ["C1", "C2", "C3", "C4", "K1", "K2", "K3", "K4", "K5", "K6"]
    ]


def test_initial_state_mini_robot_cells(mini):
    s0 = initial_state(mini)
    assert [r.held for r in s0.robots] == [(C1,), (C2,), (K1,), (K2,)]


@pytest.mark.parametrize("name", ["warehouse_15x8.yaml", "mini.yaml"])
def test_initial_state_every_robot_is_idle_with_a_full_battery(name):
    for r in initial_state(load_layout(LAYOUTS / name)).robots:
        assert r.mode is Mode.IDLE
        assert r.battery == BATTERY_CAPACITY
        assert r.order is None
        assert r.goal is None


def test_battery_capacity_matches_proposal():
    assert BATTERY_CAPACITY == 400


def test_initial_state_capacity_can_be_changed(mini):
    assert {r.battery for r in initial_state(mini, capacity=7).robots} == {7}


def test_initial_state_without_batch_has_empty_queue(mini):
    s0 = initial_state(mini)
    assert s0.queue == ()
    assert s0.delivered == 0


def test_initial_state_batch_keeps_fifo_order(mini):
    second = Order(id=2, pickup=P2, delivery=D1, created_at=0.0)
    assert initial_state(mini, batch=[ORDER, second]).queue == (ORDER, second)


def test_initial_state_accepts_any_iterable_batch(mini):
    assert initial_state(mini, batch=(o for o in [ORDER])).queue == (ORDER,)


@pytest.mark.parametrize("name", ["warehouse_15x8.yaml", "mini.yaml"])
def test_initial_state_is_hashable_and_repeatable(name):
    layout = load_layout(LAYOUTS / name)
    assert hash(initial_state(layout, batch=[ORDER])) == hash(initial_state(layout, batch=[ORDER]))
    assert initial_state(layout) == initial_state(layout)
    assert isinstance(initial_state(layout).robots, tuple)
    assert isinstance(initial_state(layout).robots[0].held, tuple)


@pytest.mark.parametrize("name", ["warehouse_15x8.yaml", "mini.yaml"])
def test_initial_state_no_two_robots_share_a_cell(name):
    held = [c for r in initial_state(load_layout(LAYOUTS / name)).robots for c in r.held]
    assert len(held) == len(set(held))


def test_initial_state_home_cells_are_taken_and_stations_free(main):
    s0 = initial_state(main)
    for cell in (*main.chargers, *main.parkings):
        assert not free(s0, cell)
    for cell in (*main.pickups, *main.deliveries):
        assert free(s0, cell)


# cheap copying


def test_helpers_do_not_change_the_state(mini_routes):
    s = system(robot(K2, (1, 2), goal=P2), robot(C1), queue=(ORDER,))
    before = copy.deepcopy(s)
    pos(s, 0)
    occ(s, K2)
    free(s, P1)
    next_cell(s, 0, mini_routes)
    rest(s, 0, mini_routes)
    assert s == before
