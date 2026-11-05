import random
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
from fleet.guards import Model, enabled_events, is_enabled
from fleet.layout import load_layout
from fleet.routes import build_routes
from fleet.state import Mode, Order, RobotState, SystemState, initial_state, next_cell
from fleet.transitions import next_state

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
THIRD = Order(id=3, pickup=P1, delivery=D1, created_at=0.0)


def robot(*held, goal=None, mode=Mode.IDLE, battery=100, order=None):
    return RobotState(held=held, mode=mode, battery=battery, order=order, goal=goal)


def system(*robots, queue=(), delivered=0):
    return SystemState(robots=robots, queue=queue, delivered=delivered)


def make_model(layout, capacity=50):
    return Model(layout=layout, routes=build_routes(layout), q_max=20, b_res=3, capacity=capacity)


def drive(s, i, model):
    """Repeat enter/arrive until robot i reaches its goal."""
    while s.robots[i].goal is not None:
        s = next_state(s, Enter(i, next_cell(s, i, model.routes)), model)
        s = next_state(s, Arrive(i), model)
    return s


@pytest.fixture(scope="module")
def main():
    return load_layout(LAYOUTS / "warehouse_15x8.yaml")


@pytest.fixture(scope="module")
def mini():
    return load_layout(LAYOUTS / "mini.yaml")


@pytest.fixture(scope="module")
def model(mini):
    return make_model(mini)


# order


def test_order_is_appended_to_an_empty_queue(model):
    s = next_state(system(robot(K1)), NewOrder(ORDER), model)
    assert s.queue == (ORDER,)


def test_order_is_appended_at_the_end(model):
    s = next_state(system(robot(K1), queue=(OTHER,)), NewOrder(ORDER), model)
    assert s.queue == (OTHER, ORDER)


def test_order_changes_nothing_else(model):
    before = system(robot(K1), delivered=2)
    s = next_state(before, NewOrder(ORDER), model)
    assert s.robots == before.robots
    assert s.delivered == 2


# assign


def test_assign_gives_the_order_to_the_robot(model):
    s = next_state(system(robot(K1), queue=(ORDER,)), Assign(0, ORDER), model)
    assert s.robots[0].order == ORDER


def test_assign_removes_only_that_order_and_keeps_fifo(model):
    before = system(robot(K1), queue=(OTHER, ORDER, THIRD))
    s = next_state(before, Assign(0, ORDER), model)
    assert s.queue == (OTHER, THIRD)


def test_assign_leaves_the_robot_idle_without_a_goal(model):
    s = next_state(system(robot(K1), queue=(ORDER,)), Assign(0, ORDER), model)
    assert s.robots[0].mode is Mode.IDLE
    assert s.robots[0].goal is None
    assert s.robots[0].held == (K1,)
    assert s.robots[0].battery == 100


# dispatch


def test_dispatch_a_to_pickup(model):
    s = next_state(system(robot(K1, order=ORDER)), Dispatch(0, P1), model)
    assert s.robots[0].mode is Mode.TO_PICK
    assert s.robots[0].goal == P1
    assert s.robots[0].order == ORDER


def test_dispatch_b_to_delivery(model):
    before = system(robot(P1, order=ORDER, mode=Mode.LOADED))
    s = next_state(before, Dispatch(0, D1), model)
    assert s.robots[0].mode is Mode.TO_DROP
    assert s.robots[0].goal == D1
    assert s.robots[0].order == ORDER


def test_dispatch_c_to_charger(model):
    s = next_state(system(robot(K1)), Dispatch(0, C1), model)
    assert s.robots[0].mode is Mode.TO_CHARGE
    assert s.robots[0].goal == C1
    assert s.robots[0].order is None


def test_dispatch_d_to_parking(model):
    s = next_state(system(robot(C1)), Dispatch(0, K2), model)
    assert s.robots[0].mode is Mode.TO_PARK
    assert s.robots[0].goal == K2
    assert s.robots[0].order is None


def test_dispatch_does_not_move_the_robot_or_use_battery(model):
    s = next_state(system(robot(K1, order=ORDER)), Dispatch(0, P1), model)
    assert s.robots[0].held == (K1,)
    assert s.robots[0].battery == 100


# enter


def test_enter_holds_both_cells_in_order(model):
    before = system(robot(K2, order=ORDER, goal=P1, mode=Mode.TO_PICK))
    s = next_state(before, Enter(0, (1, 2)), model)
    assert s.robots[0].held == (K2, (1, 2))


def test_enter_changes_nothing_else_of_the_robot(model):
    before = system(robot(K2, order=ORDER, goal=P1, mode=Mode.TO_PICK))
    s = next_state(before, Enter(0, (1, 2)), model)
    assert s.robots[0].battery == 100  # a move is paid for in arrive
    assert s.robots[0].goal == P1
    assert s.robots[0].mode is Mode.TO_PICK
    assert s.robots[0].order == ORDER


# arrive


def test_arrive_releases_the_previous_cell(model):
    before = system(robot(K2, (1, 2), order=ORDER, goal=P1, mode=Mode.TO_PICK))
    s = next_state(before, Arrive(0), model)
    assert s.robots[0].held == ((1, 2),)


def test_arrive_costs_one_unit_of_battery(model):
    before = system(robot(K2, (1, 2), order=ORDER, goal=P1, mode=Mode.TO_PICK, battery=7))
    assert next_state(before, Arrive(0), model).robots[0].battery == 6


@pytest.mark.parametrize("mode", [Mode.TO_PICK, Mode.TO_DROP, Mode.TO_CHARGE, Mode.TO_PARK])
def test_arrive_before_the_goal_keeps_goal_and_mode(model, mode):
    s = next_state(system(robot((1, 1), (2, 1), goal=D1, mode=mode)), Arrive(0), model)
    assert s.robots[0].goal == D1
    assert s.robots[0].mode is mode


def test_arrive_at_pickup_clears_goal_and_stays_to_pick(model):
    before = system(robot((1, 2), P1, order=ORDER, goal=P1, mode=Mode.TO_PICK))
    s = next_state(before, Arrive(0), model)
    assert s.robots[0].held == (P1,)
    assert s.robots[0].goal is None
    assert s.robots[0].mode is Mode.TO_PICK
    assert s.robots[0].order == ORDER


def test_arrive_at_delivery_clears_goal_and_stays_to_drop(model):
    before = system(robot((3, 1), D1, order=ORDER, goal=D1, mode=Mode.TO_DROP))
    s = next_state(before, Arrive(0), model)
    assert s.robots[0].goal is None
    assert s.robots[0].mode is Mode.TO_DROP


def test_arrive_at_charger_clears_goal_and_stays_to_charge(model):
    s = next_state(system(robot((1, 1), C1, goal=C1, mode=Mode.TO_CHARGE)), Arrive(0), model)
    assert s.robots[0].goal is None
    assert s.robots[0].mode is Mode.TO_CHARGE


def test_arrive_at_parking_clears_goal_and_becomes_idle(model):
    s = next_state(system(robot((1, 2), K2, goal=K2, mode=Mode.TO_PARK)), Arrive(0), model)
    assert s.robots[0].goal is None
    assert s.robots[0].mode is Mode.IDLE


def test_arrive_at_goal_also_costs_battery(model):
    before = system(robot((1, 2), K2, goal=K2, mode=Mode.TO_PARK, battery=7))
    assert next_state(before, Arrive(0), model).robots[0].battery == 6


# load, loaded, unload, charge


@pytest.mark.parametrize(
    ("state", "event", "mode"),
    [
        (robot(P1, order=ORDER, mode=Mode.TO_PICK), Load(0), Mode.LOADING),
        (robot(P1, order=ORDER, mode=Mode.LOADING), Loaded(0), Mode.LOADED),
        (robot(D1, order=ORDER, mode=Mode.TO_DROP), Unload(0), Mode.UNLOADING),
        (robot(C1, mode=Mode.TO_CHARGE, battery=7), Charge(0), Mode.CHARGING),
    ],
    ids=["load", "loaded", "unload", "charge"],
)
def test_mode_only_events(model, state, event, mode):
    s = next_state(system(state), event, model)
    assert s.robots[0].mode is mode
    assert s.robots[0].held == state.held
    assert s.robots[0].battery == state.battery
    assert s.robots[0].order == state.order
    assert s.robots[0].goal is None
    assert s.delivered == 0


# unloaded


def test_unloaded_finishes_the_order(model):
    before = system(robot(D1, order=ORDER, mode=Mode.UNLOADING), delivered=4)
    s = next_state(before, Unloaded(0), model)
    assert s.robots[0].mode is Mode.IDLE
    assert s.robots[0].order is None
    assert s.delivered == 5


def test_unloaded_leaves_the_robot_on_the_station(model):
    before = system(robot(D1, order=ORDER, mode=Mode.UNLOADING, battery=7))
    s = next_state(before, Unloaded(0), model)
    assert s.robots[0].held == (D1,)
    assert s.robots[0].battery == 7


# charged


def test_charged_refills_the_battery_and_makes_the_robot_idle(model):
    s = next_state(system(robot(C1, mode=Mode.CHARGING, battery=3)), Charged(0), model)
    assert s.robots[0].mode is Mode.IDLE
    assert s.robots[0].battery == model.capacity


def test_charged_uses_the_capacity_of_the_model(mini):
    model = make_model(mini, capacity=17)
    s = next_state(system(robot(C1, mode=Mode.CHARGING, battery=3)), Charged(0), model)
    assert s.robots[0].battery == 17


def test_model_capacity_defaults_to_the_battery_capacity(mini):
    model = Model(layout=mini, routes=build_routes(mini), q_max=20, b_res=3)
    assert model.capacity == 400


# general rules


CASES = [
    (robot(K2), NewOrder(ORDER)),
    (robot(K2), Assign(1, THIRD)),
    (robot(K2, order=ORDER), Dispatch(1, P1)),
    (robot(P1, order=ORDER, mode=Mode.LOADED), Dispatch(1, D1)),
    (robot(K2), Dispatch(1, C1)),
    (robot(C2), Dispatch(1, K2)),
    (robot(K2, order=ORDER, goal=P1, mode=Mode.TO_PICK), Enter(1, (1, 2))),
    (robot(K2, (1, 2), order=ORDER, goal=P1, mode=Mode.TO_PICK), Arrive(1)),
    (robot((1, 2), K2, goal=K2, mode=Mode.TO_PARK), Arrive(1)),
    (robot(P1, order=ORDER, mode=Mode.TO_PICK), Load(1)),
    (robot(P1, order=ORDER, mode=Mode.LOADING), Loaded(1)),
    (robot(D1, order=ORDER, mode=Mode.TO_DROP), Unload(1)),
    (robot(D1, order=ORDER, mode=Mode.UNLOADING), Unloaded(1)),
    (robot(C1, mode=Mode.TO_CHARGE), Charge(1)),
    (robot(C1, mode=Mode.CHARGING), Charged(1)),
]
CASE_IDS = [
    "order",
    "assign",
    "dispatch_a",
    "dispatch_b",
    "dispatch_c",
    "dispatch_d",
    "enter",
    "arrive",
    "arrive_goal",
    "load",
    "loaded",
    "unload",
    "unloaded",
    "charge",
    "charged",
]


def case_state(state):
    return system(robot(K1), state, queue=(THIRD,))


@pytest.mark.parametrize(("state", "event"), CASES, ids=CASE_IDS)
def test_other_robots_are_not_changed(model, state, event):
    before = case_state(state)
    assert next_state(before, event, model).robots[0] == before.robots[0]


@pytest.mark.parametrize(("state", "event"), CASES, ids=CASE_IDS)
def test_the_old_state_is_not_changed(model, state, event):
    before = case_state(state)
    copy = case_state(state)
    next_state(before, event, model)
    assert before == copy


@pytest.mark.parametrize(("state", "event"), CASES, ids=CASE_IDS)
def test_the_new_state_is_hashable(model, state, event):
    s = next_state(case_state(state), event, model)
    assert isinstance(hash(s), int)
    assert isinstance(s.robots, tuple)
    assert isinstance(s.queue, tuple)
    assert all(isinstance(r.held, tuple) for r in s.robots)


@pytest.mark.parametrize(
    ("state", "event"),
    [
        (system(robot(K1), queue=(OTHER,) * 20), NewOrder(ORDER)),
        (system(robot(K1)), Assign(0, ORDER)),
        (system(robot(K1)), Dispatch(0, P1)),
        (system(robot(K1, order=ORDER), robot(P1)), Dispatch(0, P1)),
        (system(robot(K2, order=ORDER, goal=P1, mode=Mode.TO_PICK)), Enter(0, (1, 1))),
        (system(robot(K1)), Arrive(0)),
        (system(robot(K1)), Load(0)),
        (system(robot(K1)), Loaded(0)),
        (system(robot(K1)), Unload(0)),
        (system(robot(K1)), Unloaded(0)),
        (system(robot(K1)), Charge(0)),
        (system(robot(K1)), Charged(0)),
    ],
    ids=lambda v: type(v).__name__ if not isinstance(v, SystemState) else "",
)
def test_event_outside_ef_raises_value_error(model, state, event):
    assert not is_enabled(state, event, model)
    with pytest.raises(ValueError):
        next_state(state, event, model)


# full life cycles


def test_full_order_life_cycle_on_mini(mini, model):
    s = initial_state(mini, capacity=50)
    i = 3  # stands on K2

    s = next_state(s, NewOrder(ORDER), model)
    assert s.queue == (ORDER,)

    s = next_state(s, Assign(i, ORDER), model)
    assert s.queue == ()
    assert s.robots[i].order == ORDER

    s = next_state(s, Dispatch(i, P1), model)
    assert s.robots[i].mode is Mode.TO_PICK

    s = drive(s, i, model)  # K2 -> P1: 2 moves
    assert s.robots[i].held == (P1,)
    assert s.robots[i].battery == 48

    s = next_state(s, Load(i), model)
    assert s.robots[i].mode is Mode.LOADING

    s = next_state(s, Loaded(i), model)
    assert s.robots[i].mode is Mode.LOADED

    s = next_state(s, Dispatch(i, D1), model)
    assert s.robots[i].mode is Mode.TO_DROP

    s = drive(s, i, model)  # P1 -> D1: 5 moves
    assert s.robots[i].held == (D1,)
    assert s.robots[i].battery == 43

    s = next_state(s, Unload(i), model)
    assert s.robots[i].mode is Mode.UNLOADING

    s = next_state(s, Unloaded(i), model)
    assert s.robots[i] == RobotState(held=(D1,), mode=Mode.IDLE, battery=43)
    assert s.queue == ()
    assert s.delivered == 1
    assert s.robots[:i] == initial_state(mini, capacity=50).robots[:i]


def test_full_order_life_cycle_on_main(main):
    model = make_model(main, capacity=400)
    s = initial_state(main, capacity=400)
    i = 4  # stands on K1
    p3, d1 = main.buffers["P3"], main.buffers["D1"]
    order = Order(id=1, pickup=p3, delivery=d1, created_at=0.0)
    for event in [NewOrder(order), Assign(i, order), Dispatch(i, p3)]:
        s = next_state(s, event, model)
    s = drive(s, i, model)  # K1 -> P3: 18 moves
    for event in [Load(i), Loaded(i), Dispatch(i, d1)]:
        s = next_state(s, event, model)
    s = drive(s, i, model)  # P3 -> D1: 10 moves
    for event in [Unload(i), Unloaded(i)]:
        s = next_state(s, event, model)
    assert s.robots[i] == RobotState(held=(d1,), mode=Mode.IDLE, battery=372)
    assert s.delivered == 1


def test_charging_cycle(mini, model):
    s = system(robot(K1, battery=20))
    s = next_state(s, Dispatch(0, C1), model)
    s = drive(s, 0, model)  # K1 -> C1: 2 moves
    assert s.robots[0].battery == 18
    s = next_state(s, Charge(0), model)
    s = next_state(s, Charged(0), model)
    assert s.robots[0] == RobotState(held=(C1,), mode=Mode.IDLE, battery=model.capacity)


def test_parking_cycle(model):
    s = system(robot(C1, battery=20))
    s = next_state(s, Dispatch(0, K1), model)
    s = drive(s, 0, model)
    assert s.robots[0].held == (K1,)
    assert s.robots[0].mode is Mode.IDLE
    assert s.robots[0].goal is None
    assert s.robots[0].battery < 20


# random runs: invariants of every reachable state


MODES_WITHOUT_GOAL = {Mode.IDLE, Mode.LOADING, Mode.LOADED, Mode.UNLOADING, Mode.CHARGING}
MODES_WITH_ORDER = {Mode.TO_PICK, Mode.LOADING, Mode.LOADED, Mode.TO_DROP, Mode.UNLOADING}
MODES_WITHOUT_ORDER = {Mode.TO_CHARGE, Mode.CHARGING, Mode.TO_PARK}


def check_invariants(s, model, created):
    held = [c for r in s.robots for c in r.held]
    assert len(held) == len(set(held)), f"S1: two robots share a cell: {s}"
    goals = [r.goal for r in s.robots if r.goal is not None]
    assert len(goals) == len(set(goals)), f"one buffer cell reserved twice: {s}"
    assert len(s.queue) <= model.q_max
    assigned = [r.order for r in s.robots if r.order is not None]
    assert len(s.queue) + len(assigned) + s.delivered == created, f"lost an order: {s}"
    for r in s.robots:
        nearest = min(model.routes.delta(r.held[0], c) for c in model.layout.chargers)
        assert r.battery >= nearest, f"S2: cannot reach a charger: {r}"
        assert r.battery >= 0, f"negative battery: {r}"
        assert r.battery <= model.capacity, f"battery above capacity: {r}"
        assert len(r.held) in (1, 2), f"wrong number of held cells: {r}"
        if len(r.held) == 2:
            assert r.held[1] in model.routes.graph[r.held[0]], f"illegal move: {r}"
        assert r.goal is None or r.goal in model.layout.buffer_cells, f"bad goal: {r}"
        if r.mode in MODES_WITHOUT_GOAL:
            assert r.goal is None, f"reservation kept in {r.mode}: {r}"
        if r.mode in MODES_WITH_ORDER:
            assert r.order is not None, f"no order in {r.mode}: {r}"
        if r.mode in MODES_WITHOUT_ORDER:
            assert r.order is None, f"unexpected order in {r.mode}: {r}"


def random_run(layout, model, seed, steps):
    """Execute random enabled events, adding a new order now and then; return the final state."""
    rng = random.Random(seed)

    def new_order(order_id):
        pickup = rng.choice(layout.pickups)
        delivery = rng.choice(layout.deliveries)
        return Order(id=order_id, pickup=pickup, delivery=delivery, created_at=0.0)

    created = 5
    batch = [new_order(order_id) for order_id in range(created)]
    s = initial_state(layout, batch=batch, capacity=model.capacity)
    for _ in range(steps):
        check_invariants(s, model, created)
        if rng.random() < 0.05 and is_enabled(s, NewOrder(new_order(created)), model):
            s = next_state(s, NewOrder(new_order(created)), model)
            created += 1
            continue
        events = enabled_events(s, model)
        if not events:
            break  # a deadlock is possible here: safe() comes in issue 7
        s = next_state(s, rng.choice(events), model)
    check_invariants(s, model, created)
    return s


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("capacity", [400, 40])
def test_random_runs_keep_invariants_on_main(main, seed, capacity):
    random_run(main, make_model(main, capacity), seed, steps=600)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("capacity", [400, 25])
def test_random_runs_keep_invariants_on_mini(mini, seed, capacity):
    random_run(mini, make_model(mini, capacity), seed, steps=600)


def test_random_runs_really_deliver_orders(main):
    # Guards against a walk that gets stuck at once and proves nothing.
    model = make_model(main, 400)
    delivered = sum(random_run(main, model, seed, steps=600).delivered for seed in range(3))
    assert delivered > 0
