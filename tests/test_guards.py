from pathlib import Path

import pytest

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
from fleet.guards import Model, enabled_events, feas, is_enabled
from fleet.layout import load_layout
from fleet.routes import build_routes
from fleet.state import Mode, Order, RobotState, SystemState, initial_state

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
OTHER = Order(id=2, pickup=P2, delivery=D1, created_at=0.0)
MOVING_MODES = [Mode.TO_PICK, Mode.TO_DROP, Mode.TO_CHARGE, Mode.TO_PARK]


def robot(*held, goal=None, mode=Mode.IDLE, battery=100, order=None):
    return RobotState(held=held, mode=mode, battery=battery, order=order, goal=goal)


def system(*robots, queue=()):
    return SystemState(robots=robots, queue=queue)


def all_modes_except(*modes):
    return [m for m in Mode if m not in modes]


@pytest.fixture(scope="module")
def main():
    return load_layout(LAYOUTS / "warehouse_15x8.yaml")


@pytest.fixture(scope="module")
def mini():
    return load_layout(LAYOUTS / "mini.yaml")


@pytest.fixture(scope="module")
def model(mini):
    return Model(layout=mini, routes=build_routes(mini), q_max=2, b_res=3)


# order


def test_order_enabled_when_queue_has_room(model):
    assert is_enabled(system(robot(K1), queue=(OTHER,)), NewOrder(ORDER), model) is True


def test_order_blocked_when_queue_is_full(model):
    s = system(robot(K1), queue=(OTHER, OTHER))
    assert is_enabled(s, NewOrder(ORDER), model) is False


# feas


def test_feas_exact_energy_limit_on_mini(model):
    # K2 -> P1: 2, P1 -> D1: 5, D1 -> farthest charger C1: 6, reserve: 3
    assert feas(system(robot(K2, battery=16)), 0, ORDER, model) is True
    assert feas(system(robot(K2, battery=15)), 0, ORDER, model) is False


def test_feas_uses_the_right_robot(model):
    s = system(robot(K1, battery=0), robot(K2, battery=16))
    assert feas(s, 0, ORDER, model) is False
    assert feas(s, 1, ORDER, model) is True


def test_feas_exact_energy_limit_on_main(main):
    # K1 -> P3: 18, P3 -> D1: 10, D1 -> farthest charger C4: 26, reserve: 5
    model = Model(layout=main, routes=build_routes(main), q_max=2, b_res=5)
    order = Order(id=1, pickup=main.buffers["P3"], delivery=main.buffers["D1"], created_at=0.0)
    k1 = main.buffers["K1"]
    assert feas(system(robot(k1, battery=59)), 0, order, model) is True
    assert feas(system(robot(k1, battery=58)), 0, order, model) is False


# assign


def test_assign_enabled(model):
    s = system(robot(K1), queue=(ORDER,))
    assert is_enabled(s, Assign(0, ORDER), model) is True


def test_assign_blocked_when_order_not_in_queue(model):
    s = system(robot(K1), queue=(OTHER,))
    assert is_enabled(s, Assign(0, ORDER), model) is False


@pytest.mark.parametrize("mode", all_modes_except(Mode.IDLE))
def test_assign_blocked_when_robot_not_idle(model, mode):
    s = system(robot(K1, mode=mode), queue=(ORDER,))
    assert is_enabled(s, Assign(0, ORDER), model) is False


def test_assign_blocked_when_robot_already_has_order(model):
    s = system(robot(K1, order=OTHER), queue=(ORDER,))
    assert is_enabled(s, Assign(0, ORDER), model) is False


def test_assign_blocked_when_robot_holds_two_cells(model):
    s = system(robot(K1, (1, 1)), queue=(ORDER,))
    assert is_enabled(s, Assign(0, ORDER), model) is False


def test_assign_blocked_when_battery_too_low(model):
    # K1 -> P1: 7 (around the one-way loop), P1 -> D1: 5, D1 -> C1: 6, reserve: 3
    enough = system(robot(K1, battery=21), queue=(ORDER,))
    too_low = system(robot(K1, battery=20), queue=(ORDER,))
    assert is_enabled(enough, Assign(0, ORDER), model) is True
    assert is_enabled(too_low, Assign(0, ORDER), model) is False


# assign only on a charger or a parking place (issue #32)


def station_test_setup(name):
    layout = load_layout(LAYOUTS / name)
    model = Model(layout=layout, routes=build_routes(layout), q_max=2, b_res=3)
    order = Order(id=1, pickup=layout.pickups[0], delivery=layout.deliveries[0], created_at=0.0)
    return layout, model, order


@pytest.mark.parametrize("name", ["warehouse_15x8.yaml", "mini.yaml"])
def test_assign_blocked_on_pickup_stations(name):
    layout, model, order = station_test_setup(name)
    for cell in layout.pickups:
        s = system(robot(cell, battery=400), queue=(order,))
        assert feas(s, 0, order, model) is True  # only the position can be the reason
        assert is_enabled(s, Assign(0, order), model) is False, cell


@pytest.mark.parametrize("name", ["warehouse_15x8.yaml", "mini.yaml"])
def test_assign_blocked_on_delivery_stations(name):
    layout, model, order = station_test_setup(name)
    for cell in layout.deliveries:
        s = system(robot(cell, battery=400), queue=(order,))
        assert feas(s, 0, order, model) is True
        assert is_enabled(s, Assign(0, order), model) is False, cell


@pytest.mark.parametrize("name", ["warehouse_15x8.yaml", "mini.yaml"])
def test_assign_enabled_on_chargers(name):
    layout, model, order = station_test_setup(name)
    for cell in layout.chargers:
        s = system(robot(cell, battery=400), queue=(order,))
        assert is_enabled(s, Assign(0, order), model) is True, cell


@pytest.mark.parametrize("name", ["warehouse_15x8.yaml", "mini.yaml"])
def test_assign_enabled_on_parking_places(name):
    layout, model, order = station_test_setup(name)
    for cell in layout.parkings:
        s = system(robot(cell, battery=400), queue=(order,))
        assert is_enabled(s, Assign(0, order), model) is True, cell


def test_assign_blocked_on_a_lane_cell(model):
    s = system(robot((1, 1), battery=400), queue=(ORDER,))
    assert is_enabled(s, Assign(0, ORDER), model) is False


def test_robot_on_a_station_can_leave_instead_of_taking_an_order(model):
    # Without the rule robot 0 could take ORDER here. Its pick-up P1 is held by robot 1, which
    # waits for D1, so neither robot could ever leave.
    to_d1 = Order(id=3, pickup=P1, delivery=D1, created_at=0.0)
    s = system(robot(D1), robot(P1, mode=Mode.LOADED, order=to_d1), queue=(ORDER,))
    events = enabled_events(s, model)
    assert [e for e in events if isinstance(e, Assign)] == []
    assert {Dispatch(0, cell) for cell in (K1, K2, C1, C2)} <= set(events)


# dispatch


def test_dispatch_a_idle_with_order_to_its_pickup(model):
    assert is_enabled(system(robot(K1, order=ORDER)), Dispatch(0, P1), model) is True


def test_dispatch_b_loaded_to_its_delivery(model):
    s = system(robot(P1, order=ORDER, mode=Mode.LOADED))
    assert is_enabled(s, Dispatch(0, D1), model) is True


def test_dispatch_c_idle_without_order_to_charger(model):
    assert is_enabled(system(robot(K1)), Dispatch(0, C1), model) is True


def test_dispatch_d_idle_without_order_to_parking(model):
    assert is_enabled(system(robot(C1)), Dispatch(0, K2), model) is True


@pytest.mark.parametrize("goal", [P2, D1, C1, K2])
def test_dispatch_idle_with_order_only_to_its_pickup(model, goal):
    assert is_enabled(system(robot(K1, order=ORDER)), Dispatch(0, goal), model) is False


@pytest.mark.parametrize("goal", [P2, C1, K2])
def test_dispatch_loaded_only_to_its_delivery(model, goal):
    s = system(robot(P1, order=ORDER, mode=Mode.LOADED))
    assert is_enabled(s, Dispatch(0, goal), model) is False


def test_dispatch_compares_cells_by_value(model):
    # Equal cells are often different tuple objects, e.g. built from another source.
    assert is_enabled(system(robot(K1, order=ORDER)), Dispatch(0, tuple(list(P1))), model) is True
    loaded = system(robot(P1, order=ORDER, mode=Mode.LOADED))
    assert is_enabled(loaded, Dispatch(0, tuple(list(D1))), model) is True


def test_dispatch_loaded_not_to_another_delivery_station(main):
    # mini has only one delivery station, so this needs the main layout.
    model = Model(layout=main, routes=build_routes(main), q_max=2, b_res=3)
    p1, d1, d2 = main.buffers["P1"], main.buffers["D1"], main.buffers["D2"]
    order = Order(id=1, pickup=p1, delivery=d1, created_at=0.0)
    s = system(robot(p1, order=order, mode=Mode.LOADED))
    assert is_enabled(s, Dispatch(0, d1), model) is True
    assert is_enabled(s, Dispatch(0, d2), model) is False


@pytest.mark.parametrize("goal", [P1, D1])
def test_dispatch_idle_without_order_not_to_stations(model, goal):
    assert is_enabled(system(robot(K1)), Dispatch(0, goal), model) is False


@pytest.mark.parametrize("mode", all_modes_except(Mode.IDLE, Mode.LOADED))
def test_dispatch_blocked_in_other_modes(model, mode):
    with_order = system(robot(K1, order=ORDER, mode=mode))
    without_order = system(robot(K1, mode=mode))
    assert is_enabled(with_order, Dispatch(0, P1), model) is False
    assert is_enabled(with_order, Dispatch(0, D1), model) is False
    assert is_enabled(without_order, Dispatch(0, C1), model) is False
    assert is_enabled(without_order, Dispatch(0, K2), model) is False


def test_dispatch_blocked_when_robot_holds_two_cells(model):
    s = system(robot(K1, (1, 1)))
    assert is_enabled(s, Dispatch(0, C1), model) is False


def test_dispatch_blocked_when_robot_stands_on_a_lane(model):
    assert is_enabled(system(robot((1, 1))), Dispatch(0, C1), model) is False


def test_dispatch_blocked_when_goal_is_occupied(model):
    s = system(robot(K1, order=ORDER), robot(P1))
    assert is_enabled(s, Dispatch(0, P1), model) is False


def test_dispatch_blocked_when_goal_is_reserved(model):
    s = system(robot(K1, order=ORDER), robot(K2, goal=P1, mode=Mode.TO_PARK))
    assert is_enabled(s, Dispatch(0, P1), model) is False


def test_dispatch_blocked_to_the_cell_it_stands_on(model):
    assert is_enabled(system(robot(C1)), Dispatch(0, C1), model) is False


# dispatch: energy (S2), not in Table 2


def test_dispatch_c_needs_energy_to_reach_that_charger(model):
    # K1 -> C1: 2 moves
    assert is_enabled(system(robot(K1, battery=2)), Dispatch(0, C1), model) is True
    assert is_enabled(system(robot(K1, battery=1)), Dispatch(0, C1), model) is False


def test_dispatch_c_counts_the_chosen_charger_not_the_nearest(model):
    # K1 -> C2: 3 moves, although C1 is only 2 moves away
    assert is_enabled(system(robot(K1, battery=3)), Dispatch(0, C2), model) is True
    assert is_enabled(system(robot(K1, battery=2)), Dispatch(0, C2), model) is False


def test_dispatch_d_needs_energy_to_park_and_then_reach_the_nearest_charger(model):
    # C2 -> K2: 6 moves, K2 -> nearest charger C1: 3 moves (the farther C2 is 4 moves away)
    assert is_enabled(system(robot(C2, battery=9)), Dispatch(0, K2), model) is True
    assert is_enabled(system(robot(C2, battery=8)), Dispatch(0, K2), model) is False


def test_dispatch_energy_check_does_not_use_the_reserve(mini):
    model = Model(layout=mini, routes=build_routes(mini), q_max=2, b_res=50)
    assert is_enabled(system(robot(K1, battery=2)), Dispatch(0, C1), model) is True
    assert is_enabled(system(robot(C2, battery=9)), Dispatch(0, K2), model) is True


def test_dispatch_with_an_order_has_no_extra_energy_check(model):
    # feas() has already checked the whole trip in assign.
    assert is_enabled(system(robot(K1, order=ORDER, battery=1)), Dispatch(0, P1), model) is True
    loaded = system(robot(P1, order=ORDER, mode=Mode.LOADED, battery=1))
    assert is_enabled(loaded, Dispatch(0, D1), model) is True


def test_enabled_events_low_battery_robot_can_only_go_to_a_reachable_charger(model):
    # K1 -> C1: 2, K1 -> C2: 3, K1 -> K2 and then a charger: 7 + 3
    assert enabled_events(system(robot(K1, battery=2)), model) == [Dispatch(0, C1)]


# enter


@pytest.mark.parametrize("mode", MOVING_MODES)
def test_enter_enabled_in_every_moving_mode(model, mode):
    s = system(robot(K2, goal=P1, mode=mode))
    assert is_enabled(s, Enter(0, (1, 2)), model) is True


@pytest.mark.parametrize("mode", all_modes_except(*MOVING_MODES))
def test_enter_blocked_in_other_modes(model, mode):
    s = system(robot(K2, goal=P1, mode=mode))
    assert is_enabled(s, Enter(0, (1, 2)), model) is False


def test_enter_blocked_without_goal(model):
    # The robot has reached its pick-up station: mode is still ToPick, the goal is cleared.
    s = system(robot(P1, order=ORDER, mode=Mode.TO_PICK))
    assert is_enabled(s, Enter(0, (1, 2)), model) is False


def test_enter_blocked_when_already_moving(model):
    s = system(robot(K2, (1, 2), goal=P1, mode=Mode.TO_PICK))
    assert is_enabled(s, Enter(0, (1, 2)), model) is False
    assert is_enabled(s, Enter(0, P1), model) is False


def test_enter_blocked_when_holding_two_cells_even_if_next_cell_is_free(model):
    # Route from (2, 2) to P2 goes through (2, 1), which is free here.
    s = system(robot((2, 2), (1, 2), goal=P2, mode=Mode.TO_PICK))
    assert is_enabled(s, Enter(0, (2, 1)), model) is False


def test_enter_only_into_the_next_cell_of_the_route(model):
    s = system(robot((2, 2), goal=P2, mode=Mode.TO_PICK))
    assert is_enabled(s, Enter(0, (2, 1)), model) is True
    assert is_enabled(s, Enter(0, (1, 2)), model) is False


def test_enter_next_cell_depends_on_the_goal(model):
    s = system(robot((2, 2), goal=P1, mode=Mode.TO_PICK))
    assert is_enabled(s, Enter(0, (1, 2)), model) is True
    assert is_enabled(s, Enter(0, (2, 1)), model) is False


def test_enter_blocked_when_next_cell_has_a_standing_robot(model):
    s = system(robot(K2, goal=P1, mode=Mode.TO_PICK), robot((1, 2)))
    assert is_enabled(s, Enter(0, (1, 2)), model) is False


def test_enter_blocked_when_next_cell_is_held_by_a_moving_robot(model):
    s = system(robot(K2, goal=P1, mode=Mode.TO_PICK), robot((2, 2), (1, 2)))
    assert is_enabled(s, Enter(0, (1, 2)), model) is False


def test_enter_own_reserved_goal(model):
    # The goal is reserved by this robot, so free(goal) is False, but it may still enter.
    s = system(robot((3, 2), goal=P2, mode=Mode.TO_PICK))
    assert is_enabled(s, Enter(0, P2), model) is True


def test_enter_needs_one_unit_of_battery(model):
    one = system(robot(K2, goal=P1, mode=Mode.TO_PICK, battery=1))
    zero = system(robot(K2, goal=P1, mode=Mode.TO_PICK, battery=0))
    assert is_enabled(one, Enter(0, (1, 2)), model) is True
    assert is_enabled(zero, Enter(0, (1, 2)), model) is False


# arrive


def test_arrive_enabled_while_moving(model):
    s = system(robot(K2, (1, 2), goal=P1, mode=Mode.TO_PICK))
    assert is_enabled(s, Arrive(0), model) is True


def test_arrive_blocked_when_standing(model):
    s = system(robot(K2, goal=P1, mode=Mode.TO_PICK))
    assert is_enabled(s, Arrive(0), model) is False


# load, loaded


def test_load_enabled_at_the_pickup_station(model):
    s = system(robot(P1, order=ORDER, mode=Mode.TO_PICK))
    assert is_enabled(s, Load(0), model) is True


@pytest.mark.parametrize("mode", all_modes_except(Mode.TO_PICK))
def test_load_blocked_in_other_modes(model, mode):
    s = system(robot(P1, order=ORDER, mode=mode))
    assert is_enabled(s, Load(0), model) is False


def test_load_blocked_before_reaching_the_station(model):
    standing = system(robot((1, 2), order=ORDER, goal=P1, mode=Mode.TO_PICK))
    moving = system(robot((1, 2), P1, order=ORDER, goal=P1, mode=Mode.TO_PICK))
    assert is_enabled(standing, Load(0), model) is False
    assert is_enabled(moving, Load(0), model) is False


def test_load_blocked_at_another_station(model):
    s = system(robot(P2, order=ORDER, mode=Mode.TO_PICK))
    assert is_enabled(s, Load(0), model) is False


def test_load_blocked_while_holding_two_cells(model):
    # The robot is on its station but already drives out, so it does not stand there.
    s = system(robot(P1, (1, 2), order=ORDER, goal=P2, mode=Mode.TO_PICK))
    assert is_enabled(s, Load(0), model) is False


def test_loaded_enabled_while_loading(model):
    s = system(robot(P1, order=ORDER, mode=Mode.LOADING))
    assert is_enabled(s, Loaded(0), model) is True


@pytest.mark.parametrize("mode", all_modes_except(Mode.LOADING))
def test_loaded_blocked_in_other_modes(model, mode):
    s = system(robot(P1, order=ORDER, mode=mode))
    assert is_enabled(s, Loaded(0), model) is False


# unload, unloaded


def test_unload_enabled_at_the_delivery_station(model):
    s = system(robot(D1, order=ORDER, mode=Mode.TO_DROP))
    assert is_enabled(s, Unload(0), model) is True


@pytest.mark.parametrize("mode", all_modes_except(Mode.TO_DROP))
def test_unload_blocked_in_other_modes(model, mode):
    s = system(robot(D1, order=ORDER, mode=mode))
    assert is_enabled(s, Unload(0), model) is False


def test_unload_blocked_before_reaching_the_station(model):
    standing = system(robot((3, 1), order=ORDER, goal=D1, mode=Mode.TO_DROP))
    moving = system(robot((3, 1), D1, order=ORDER, goal=D1, mode=Mode.TO_DROP))
    assert is_enabled(standing, Unload(0), model) is False
    assert is_enabled(moving, Unload(0), model) is False


def test_unload_blocked_at_another_station(main):
    # mini has only one delivery station, so this needs the main layout.
    model = Model(layout=main, routes=build_routes(main), q_max=2, b_res=3)
    p1, d1, d2 = main.buffers["P1"], main.buffers["D1"], main.buffers["D2"]
    order = Order(id=1, pickup=p1, delivery=d1, created_at=0.0)
    assert is_enabled(system(robot(d1, order=order, mode=Mode.TO_DROP)), Unload(0), model) is True
    assert is_enabled(system(robot(d2, order=order, mode=Mode.TO_DROP)), Unload(0), model) is False


def test_unload_blocked_while_holding_two_cells(model):
    s = system(robot(D1, (3, 1), order=ORDER, goal=C1, mode=Mode.TO_DROP))
    assert is_enabled(s, Unload(0), model) is False


def test_unloaded_enabled_while_unloading(model):
    s = system(robot(D1, order=ORDER, mode=Mode.UNLOADING))
    assert is_enabled(s, Unloaded(0), model) is True


@pytest.mark.parametrize("mode", all_modes_except(Mode.UNLOADING))
def test_unloaded_blocked_in_other_modes(model, mode):
    s = system(robot(D1, order=ORDER, mode=mode))
    assert is_enabled(s, Unloaded(0), model) is False


# charge, charged


def test_charge_enabled_on_a_charger(model):
    assert is_enabled(system(robot(C1, mode=Mode.TO_CHARGE)), Charge(0), model) is True


@pytest.mark.parametrize("mode", all_modes_except(Mode.TO_CHARGE))
def test_charge_blocked_in_other_modes(model, mode):
    assert is_enabled(system(robot(C1, mode=mode)), Charge(0), model) is False


def test_charge_blocked_outside_a_charger(model):
    lane = system(robot((1, 1), goal=C1, mode=Mode.TO_CHARGE))
    moving = system(robot((1, 1), C1, goal=C1, mode=Mode.TO_CHARGE))
    parking = system(robot(K1, mode=Mode.TO_CHARGE))
    assert is_enabled(lane, Charge(0), model) is False
    assert is_enabled(moving, Charge(0), model) is False
    assert is_enabled(parking, Charge(0), model) is False


def test_charge_blocked_while_holding_two_cells(model):
    arriving = system(robot((1, 1), C1, mode=Mode.TO_CHARGE))
    leaving = system(robot(C1, (1, 1), mode=Mode.TO_CHARGE))
    assert is_enabled(arriving, Charge(0), model) is False
    assert is_enabled(leaving, Charge(0), model) is False


def test_charge_blocked_on_the_start_charger_when_heading_to_another(model):
    # Not in Table 2: charging here would keep the reservation of C2 forever.
    s = system(robot(C1, goal=C2, mode=Mode.TO_CHARGE))
    assert is_enabled(s, Charge(0), model) is False


def test_charged_enabled_while_charging(model):
    assert is_enabled(system(robot(C1, mode=Mode.CHARGING)), Charged(0), model) is True


@pytest.mark.parametrize("mode", all_modes_except(Mode.CHARGING))
def test_charged_blocked_in_other_modes(model, mode):
    assert is_enabled(system(robot(C1, mode=mode)), Charged(0), model) is False


# events of one robot do not depend on the index of another


def test_guards_use_the_robot_from_the_event(model):
    s = system(robot(K1), robot(P1, order=ORDER, mode=Mode.TO_PICK))
    assert is_enabled(s, Load(0), model) is False
    assert is_enabled(s, Load(1), model) is True


# enabled_events


def test_enabled_events_returns_a_list(model):
    assert isinstance(enabled_events(system(robot(K1)), model), list)


def test_enabled_events_nothing_in_s0_without_orders(mini, model):
    # Every home cell is taken, so no robot has anywhere to go.
    assert enabled_events(initial_state(mini), model) == []


def test_enabled_events_every_robot_can_take_the_order_in_s0(mini, model):
    events = enabled_events(initial_state(mini, batch=[ORDER]), model)
    assert sorted(e.robot for e in events) == [0, 1, 2, 3]
    assert set(events) == {Assign(i, ORDER) for i in range(4)}


def test_enabled_events_skips_robot_with_too_low_battery(mini, model):
    events = enabled_events(initial_state(mini, batch=[ORDER], capacity=1), model)
    assert events == []


def test_enabled_events_one_assign_per_queued_order(model):
    s = system(robot(K1), queue=(ORDER, OTHER))
    assigns = {e for e in enabled_events(s, model) if isinstance(e, Assign)}
    assert assigns == {Assign(0, ORDER), Assign(0, OTHER)}


def test_enabled_events_idle_robot_can_go_to_every_free_home_cell(model):
    s = system(robot(K1))
    assert set(enabled_events(s, model)) == {Dispatch(0, C1), Dispatch(0, C2), Dispatch(0, K2)}


def test_enabled_events_robot_with_order_goes_only_to_pickup(model):
    s = system(robot(K1, order=ORDER))
    assert set(enabled_events(s, model)) == {Dispatch(0, P1)}


def test_enabled_events_loaded_robot_goes_only_to_delivery(model):
    s = system(robot(P1, order=ORDER, mode=Mode.LOADED))
    assert set(enabled_events(s, model)) == {Dispatch(0, D1)}


def test_enabled_events_driving_robot_enters_only_the_next_cell(model):
    s = system(robot((2, 2), order=OTHER, goal=P2, mode=Mode.TO_PICK))
    assert set(enabled_events(s, model)) == {Enter(0, (2, 1))}


def test_enabled_events_blocked_robot_has_no_events(model):
    s = system(robot((2, 2), order=OTHER, goal=P2, mode=Mode.TO_PICK), robot((2, 1)))
    assert enabled_events(s, model) == []


def test_enabled_events_moving_robot_only_arrives(model):
    s = system(robot(K2, (1, 2), order=ORDER, goal=P1, mode=Mode.TO_PICK))
    assert enabled_events(s, model) == [Arrive(0)]


@pytest.mark.parametrize(
    ("state", "event"),
    [
        (robot(P1, order=ORDER, mode=Mode.TO_PICK), Load(0)),
        (robot(P1, order=ORDER, mode=Mode.LOADING), Loaded(0)),
        (robot(D1, order=ORDER, mode=Mode.TO_DROP), Unload(0)),
        (robot(D1, order=ORDER, mode=Mode.UNLOADING), Unloaded(0)),
        (robot(C1, mode=Mode.TO_CHARGE), Charge(0)),
        (robot(C1, mode=Mode.CHARGING), Charged(0)),
    ],
    ids=lambda v: type(v).__name__ if not isinstance(v, RobotState) else "",
)
def test_enabled_events_station_events(model, state, event):
    assert enabled_events(system(state), model) == [event]


def test_enabled_events_several_robots_at_once(model):
    s = system(
        robot(K2, (1, 2), order=ORDER, goal=P1, mode=Mode.TO_PICK),
        robot(D1, order=OTHER, mode=Mode.UNLOADING),
        robot(C1, mode=Mode.CHARGING),
    )
    assert set(enabled_events(s, model)) == {Arrive(0), Unloaded(1), Charged(2)}


def test_enabled_events_never_lists_new_order(model):
    events = enabled_events(system(robot(K1), queue=(ORDER,)), model)
    assert not any(isinstance(e, NewOrder) for e in events)


STATES = [
    system(robot(K1)),
    system(robot(K1), robot(K2, goal=P1, mode=Mode.TO_PARK), queue=(ORDER, OTHER)),
    system(robot(K1, order=ORDER), robot(P2, order=OTHER, mode=Mode.LOADED)),
    system(
        robot((2, 2), order=OTHER, goal=P2, mode=Mode.TO_PICK),
        robot((2, 1)),
        robot(K1, (1, 1), goal=K2, mode=Mode.TO_PARK),
    ),
    system(
        robot((3, 2), P2, order=OTHER, goal=P2, mode=Mode.TO_PICK),
        robot(C1, goal=C2, mode=Mode.TO_CHARGE),
    ),
    system(
        robot(K2, order=ORDER, goal=P1, mode=Mode.TO_PICK, battery=0),
        robot(C1, mode=Mode.CHARGING),
    ),
]


@pytest.mark.parametrize("s", STATES)
def test_enabled_events_agrees_with_is_enabled(mini, model, s):
    candidates = []
    for i in range(len(s.robots)):
        candidates += [Arrive(i), Load(i), Loaded(i), Unload(i), Unloaded(i), Charge(i), Charged(i)]
        candidates += [Assign(i, o) for o in s.queue]
        candidates += [Dispatch(i, g) for g in mini.buffer_cells]
        candidates += [Enter(i, v) for v in mini.floor]
    expected = {e for e in candidates if is_enabled(s, e, model)}
    events = enabled_events(s, model)
    assert set(events) == expected
    assert len(events) == len(set(events)), "an event is listed twice"
