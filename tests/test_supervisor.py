import ast
import random
from pathlib import Path

import pytest

import fleet.supervisor
from fleet.events import (
    CONTROLLABLE,
    UNCONTROLLABLE,
    Arrive,
    Assign,
    Charge,
    Charged,
    Dispatch,
    Enter,
    Load,
    Unload,
)
from fleet.guards import Model, enabled_events, is_enabled, nearest_charging_dist
from fleet.layout import load_layout
from fleet.routes import build_routes
from fleet.safety import is_safe_state, safe
from fleet.state import Mode, Order, RobotState, SystemState, initial_state, pos
from fleet.supervisor import BaselinePolicy, Supervisor
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


def robot(*held, goal=None, mode=Mode.IDLE, battery=100, order=None):
    return RobotState(held=held, mode=mode, battery=battery, order=order, goal=goal)


def system(*robots, queue=()):
    return SystemState(robots=robots, queue=queue)


def make_model(layout, capacity=400):
    return Model(layout=layout, routes=build_routes(layout), q_max=20, b_res=3, capacity=capacity)


@pytest.fixture(scope="module")
def mini():
    return load_layout(LAYOUTS / "mini.yaml")


@pytest.fixture(scope="module")
def main():
    return load_layout(LAYOUTS / "warehouse_15x8.yaml")


@pytest.fixture(scope="module")
def model(mini):
    return make_model(mini)


def almost_closed_loop():
    """Robot 3 would close the 4-cell loop by entering (2, 1); robot 0 would open it."""
    return system(
        robot((2, 2), goal=C2, mode=Mode.TO_CHARGE),
        robot((3, 1), order=OTHER, goal=P2, mode=Mode.TO_PICK),
        robot((3, 2), order=ORDER, goal=P1, mode=Mode.TO_PICK),
        robot((1, 1), order=ORDER, goal=D1, mode=Mode.TO_DROP),
    )


def safe_alone_but_not_together():
    """Enter(0, (2, 1)) and Enter(2, (1, 2)) are each safe, but together they close a cycle.

    Robot 0 would need (1, 2) held by robot 2, robot 2 would need (1, 1) held by robot 1, and
    robot 1 would need (2, 1) held by robot 0.
    """
    return system(
        robot(C2, order=ORDER, goal=P1, mode=Mode.TO_PICK),
        robot((1, 1), order=OTHER, goal=D1, mode=Mode.TO_DROP),
        robot(K2, goal=C1, mode=Mode.TO_CHARGE),
    )


class FirstPolicy:
    """Takes the first candidate each time."""

    def choose(self, s, candidates, model):
        return candidates[0] if candidates else None


class LastPolicy:
    """Takes the last candidate each time."""

    def choose(self, s, candidates, model):
        return candidates[-1] if candidates else None


class NothingPolicy:
    def choose(self, s, candidates, model):
        return None


class OncePolicy:
    """Chooses the given event once, then nothing."""

    def __init__(self, event):
        self.event = event

    def choose(self, s, candidates, model):
        event, self.event = self.event, None
        return event


class RecordingPolicy:
    """Works like FirstPolicy and remembers what it was shown."""

    def __init__(self):
        self.calls = []

    def choose(self, s, candidates, model):
        self.calls.append((s, list(candidates)))
        return candidates[0] if candidates else None


def assert_valid_sequence(s, events, model):
    """Each event must be controllable, enabled and safe at the moment it is applied."""
    for e in events:
        assert type(e) in CONTROLLABLE, e
        assert is_enabled(s, e, model), e
        assert safe(s, e, model), e
        s = next_state(s, e, model)
    return s


# candidates


def test_candidates_contain_only_controllable_events(model):
    s = system(
        robot(K1, (1, 1), goal=C1, mode=Mode.TO_CHARGE),  # Arrive is enabled
        robot(P2, mode=Mode.LOADING, order=OTHER),  # Loaded is enabled
        robot(D1, mode=Mode.UNLOADING, order=ORDER),  # Unloaded is enabled
        robot(C1, mode=Mode.CHARGING),  # Charged is enabled
    )
    assert {type(e) for e in enabled_events(s, model)} & UNCONTROLLABLE
    assert Supervisor(model, FirstPolicy()).candidates(s) == []


def test_candidates_leave_out_unsafe_moves(model):
    s = almost_closed_loop()
    assert Enter(3, (2, 1)) in enabled_events(s, model)
    assert Enter(3, (2, 1)) not in Supervisor(model, FirstPolicy()).candidates(s)


def test_candidates_keep_safe_moves(model):
    assert Enter(0, (2, 1)) in Supervisor(model, FirstPolicy()).candidates(almost_closed_loop())


def test_candidates_include_every_kind_of_controllable_event(model):
    s = system(
        robot(K1),
        robot(K2, order=OTHER),
        robot(P1, mode=Mode.TO_PICK, order=ORDER),
        robot(C1, mode=Mode.TO_CHARGE),
        robot(D1, mode=Mode.TO_DROP, order=ORDER),
        queue=(ORDER,),
    )
    kinds = {type(e) for e in Supervisor(model, FirstPolicy()).candidates(s)}
    assert {Assign, Dispatch, Load, Unload, Charge} <= kinds


# decide: the loop


def test_policy_that_wants_nothing_gives_no_events(model):
    s = system(robot(K1), queue=(ORDER,))
    assert Supervisor(model, NothingPolicy()).decide(s) == []


def test_decide_runs_the_chain_of_events_in_one_call(model):
    # Without this the order would wait forever: no uncontrollable event comes next to wake it up.
    s = system(robot(K1), queue=(ORDER,))
    events = Supervisor(model, FirstPolicy()).decide(s)
    assert events == [Assign(0, ORDER), Dispatch(0, P1), Enter(0, (1, 1))]


def test_decide_shows_the_policy_the_state_after_each_event(model):
    s = system(robot(K1), queue=(ORDER,))
    policy = RecordingPolicy()
    events = Supervisor(model, policy).decide(s)
    assert len(policy.calls) == len(events) + 1
    current = s
    for (seen, _), e in zip(policy.calls, events, strict=False):
        assert seen == current
        current = next_state(current, e, model)
    assert policy.calls[-1][0] == current


def test_decide_asks_the_policy_even_with_no_candidates(model):
    s = system(robot(K1, (1, 1), goal=C1, mode=Mode.TO_CHARGE))
    policy = RecordingPolicy()
    assert Supervisor(model, policy).decide(s) == []
    assert policy.calls == [(s, [])]


def test_decide_rejects_an_unsafe_choice(model):
    s = almost_closed_loop()
    assert is_enabled(s, Enter(3, (2, 1)), model)
    with pytest.raises(ValueError):
        Supervisor(model, OncePolicy(Enter(3, (2, 1)))).decide(s)


def test_decide_rejects_an_uncontrollable_choice(model):
    s = system(robot(K1, (1, 1), goal=C1, mode=Mode.TO_CHARGE))
    assert is_enabled(s, Arrive(0), model)
    with pytest.raises(ValueError):
        Supervisor(model, OncePolicy(Arrive(0))).decide(s)


def test_decide_rejects_a_choice_that_is_not_enabled(model):
    s = system(robot(K1))
    with pytest.raises(ValueError):
        Supervisor(model, OncePolicy(Dispatch(0, P2))).decide(s)


# decide: conflicts


@pytest.mark.parametrize("policy", [FirstPolicy, LastPolicy])
def test_one_robot_per_cell(model, policy):
    s = system(
        robot((1, 1), goal=C2, mode=Mode.TO_CHARGE),
        robot((2, 2), order=OTHER, goal=P2, mode=Mode.TO_PICK),
    )
    assert {Enter(0, (2, 1)), Enter(1, (2, 1))} <= set(enabled_events(s, model))
    events = Supervisor(model, policy()).decide(s)
    assert len([e for e in events if isinstance(e, Enter) and e.cell == (2, 1)]) == 1
    assert_valid_sequence(s, events, model)


class AssignPolicy:
    """Gives orders only, to the robot with the given position on the list of candidates."""

    def __init__(self, position):
        self.position = position

    def choose(self, s, candidates, model):
        assigns = [e for e in candidates if isinstance(e, Assign)]
        return assigns[self.position] if assigns else None


@pytest.mark.parametrize("position", [0, -1])
def test_one_robot_per_order(model, position):
    s = system(robot(K1), robot(K2), queue=(ORDER,))
    events = Supervisor(model, AssignPolicy(position)).decide(s)
    assert len(events) == 1
    assert_valid_sequence(s, events, model)


class ChargerPolicy:
    """Sends robots to the charger C1 only."""

    def choose(self, s, candidates, model):
        chosen = [e for e in candidates if isinstance(e, Dispatch) and e.goal == C1]
        return chosen[0] if chosen else None


def test_one_robot_per_charger(model):
    s = system(robot(K1), robot(K2))
    candidates = Supervisor(model, FirstPolicy()).candidates(s)
    assert {Dispatch(0, C1), Dispatch(1, C1)} <= set(candidates)
    assert Supervisor(model, ChargerPolicy()).decide(s) == [Dispatch(0, C1)]


@pytest.mark.parametrize("policy", [FirstPolicy, LastPolicy, lambda: BaselinePolicy(b_low=30)])
def test_moves_safe_alone_are_not_issued_together(model, policy):
    s = safe_alone_but_not_together()
    first, second = Enter(0, (2, 1)), Enter(2, (1, 2))
    candidates = Supervisor(model, FirstPolicy()).candidates(s)
    assert first in candidates
    assert second in candidates

    events = Supervisor(model, policy()).decide(s)
    assert not {first, second} <= set(events)
    assert is_safe_state(assert_valid_sequence(s, events, model), model)


# baseline: operations and orders


def baseline_events(s, model, b_low=30, seed=0):
    return Supervisor(model, BaselinePolicy(b_low=b_low, seed=seed)).decide(s)


@pytest.mark.parametrize(
    ("r", "event"),
    [
        (robot(P1, mode=Mode.TO_PICK, order=ORDER), Load(0)),
        (robot(D1, mode=Mode.TO_DROP, order=ORDER), Unload(0)),
        (robot(C1, mode=Mode.TO_CHARGE), Charge(0)),
    ],
    ids=["load", "unload", "charge"],
)
def test_baseline_starts_operations(model, r, event):
    assert baseline_events(system(r), model) == [event]


def test_baseline_sends_a_loaded_robot_to_its_delivery(model):
    events = baseline_events(system(robot(P1, mode=Mode.LOADED, order=ORDER)), model)
    assert events == [Dispatch(0, D1), Enter(0, (1, 2))]


def test_baseline_loaded_robot_waits_for_a_taken_delivery(model):
    s = system(
        robot(P1, mode=Mode.LOADED, order=ORDER), robot(D1, mode=Mode.UNLOADING, order=OTHER)
    )
    assert baseline_events(s, model) == []


def test_baseline_assigns_the_oldest_order_first(model):
    events = baseline_events(system(robot(K1), queue=(OTHER, ORDER)), model)
    assert Assign(0, OTHER) in events
    assert Assign(0, ORDER) not in events


def test_baseline_sends_the_robot_to_the_pickup_of_its_order(model):
    events = baseline_events(system(robot(K1), queue=(ORDER,)), model)
    assert events[:2] == [Assign(0, ORDER), Dispatch(0, P1)]


def test_baseline_robot_with_order_waits_for_a_taken_pickup(model):
    s = system(robot(K1, order=ORDER), robot(P1, mode=Mode.LOADING, order=OTHER))
    assert baseline_events(s, model) == []


def chosen_robot(model, seed):
    s = system(robot(K1), robot(K2), robot(C1), robot(C2), queue=(ORDER,))
    assigns = [e for e in baseline_events(s, model, seed=seed) if isinstance(e, Assign)]
    assert len(assigns) == 1
    return assigns[0].robot


def test_baseline_chooses_a_random_robot_for_an_order(model):
    assert {chosen_robot(model, seed) for seed in range(30)} == {0, 1, 2, 3}


def test_baseline_same_seed_gives_the_same_robot(model):
    for seed in range(5):
        assert chosen_robot(model, seed) == chosen_robot(model, seed)


def test_baseline_does_not_give_orders_to_robots_with_low_battery(model):
    # Battery 25 passes feas (needs 21), but it is below b_low, so the robot goes to charge.
    s = system(robot(K1, battery=25), queue=(ORDER,))
    events = baseline_events(s, model, b_low=30)
    assert [e for e in events if isinstance(e, Assign)] == []
    assert events[0] in (Dispatch(0, C1), Dispatch(0, C2))


# baseline: idle robots


def test_baseline_idle_robots_at_home_stay_there(model):
    s = system(robot(K1), robot(K2), robot(C1), robot(C2))
    assert baseline_events(s, model) == []


def test_baseline_clears_a_station_to_a_parking_place(model):
    events = baseline_events(system(robot(D1)), model)
    assert events[0] in (Dispatch(0, K1), Dispatch(0, K2))


def test_baseline_clears_a_station_to_a_charger_if_parking_is_full(model):
    events = baseline_events(system(robot(D1), robot(K1), robot(K2)), model)
    assert events[0] in (Dispatch(0, C1), Dispatch(0, C2))


def test_baseline_sends_a_low_robot_from_a_station_to_a_charger(model):
    events = baseline_events(system(robot(D1, battery=20)), model)
    assert events[0] in (Dispatch(0, C1), Dispatch(0, C2))


def test_baseline_low_robot_leaves_a_station_for_parking_if_chargers_are_full(model):
    events = baseline_events(system(robot(D1, battery=20), robot(C1), robot(C2)), model)
    assert events[0] in (Dispatch(0, K1), Dispatch(0, K2))


def test_baseline_sends_a_low_parked_robot_to_a_charger(model):
    events = baseline_events(system(robot(K1, battery=20)), model)
    assert events[0] in (Dispatch(0, C1), Dispatch(0, C2))


def test_baseline_low_threshold_is_exclusive(model):
    assert baseline_events(system(robot(K1, battery=30)), model, b_low=30) == []
    assert baseline_events(system(robot(K1, battery=29)), model, b_low=30) != []


# baseline: first-come-first-served


def fcfs_states(first):
    """Robots 0 and 1 both want (2, 1), held by robot 2. `first` starts waiting before the other."""
    waiting = {
        0: robot((1, 1), goal=D1, mode=Mode.TO_DROP, order=ORDER),
        1: robot((2, 2), goal=C2, mode=Mode.TO_CHARGE),
    }
    coming = {
        0: robot((1, 2), (1, 1), goal=D1, mode=Mode.TO_DROP, order=ORDER),
        1: robot((3, 2), (2, 2), goal=C2, mode=Mode.TO_CHARGE),
    }
    blocker_moving = robot((2, 1), (3, 1), goal=P2, mode=Mode.TO_PICK, order=OTHER)
    blocker_gone = robot((3, 1), goal=P2, mode=Mode.TO_PICK, order=OTHER)
    other = 1 - first
    earlier = {first: waiting[first], other: coming[other]}
    return [
        system(earlier[0], earlier[1], blocker_moving),
        system(waiting[0], waiting[1], blocker_moving),
        system(waiting[0], waiting[1], blocker_gone),
    ]


@pytest.mark.parametrize("first", [0, 1])
def test_baseline_first_come_first_served(model, first):
    supervisor = Supervisor(model, BaselinePolicy(b_low=30))
    states = fcfs_states(first)
    assert supervisor.decide(states[0]) == []
    assert supervisor.decide(states[1]) == []

    events = supervisor.decide(states[2])
    assert Enter(first, (2, 1)) in events
    assert Enter(1 - first, (2, 1)) not in events


def test_baseline_tie_goes_to_the_lowest_robot_number(model):
    # Both robots start waiting in the same state.
    s = fcfs_states(first=0)[2]
    events = baseline_events(s, model)
    assert Enter(0, (2, 1)) in events
    assert Enter(1, (2, 1)) not in events


def test_baseline_waiting_restarts_on_each_new_cell(model):
    # Robot 1 started waiting first, but on the previous cell. On (2, 2) it comes after robot 0.
    supervisor = Supervisor(model, BaselinePolicy(b_low=30))
    r0_coming = robot((1, 2), (1, 1), goal=P2, mode=Mode.TO_PICK, order=OTHER)
    r0_waiting = robot((1, 1), goal=P2, mode=Mode.TO_PICK, order=OTHER)
    r1_on_old_cell = robot((3, 2), goal=D1, mode=Mode.TO_DROP, order=ORDER)
    r1_moving = robot((3, 2), (2, 2), goal=D1, mode=Mode.TO_DROP, order=ORDER)
    r1_on_new_cell = robot((2, 2), goal=D1, mode=Mode.TO_DROP, order=ORDER)
    r2_near = robot((2, 2), (2, 1), goal=C2, mode=Mode.TO_CHARGE)
    r2_leaving = robot((2, 1), C2, goal=C2, mode=Mode.TO_CHARGE)
    r2_done = robot(C2, mode=Mode.TO_CHARGE)

    assert supervisor.decide(system(r0_coming, r1_on_old_cell, r2_near)) == []
    assert supervisor.decide(system(r0_waiting, r1_on_old_cell, r2_near)) == []
    assert supervisor.decide(system(r0_waiting, r1_moving, r2_leaving)) == []
    assert supervisor.decide(system(r0_waiting, r1_on_new_cell, r2_leaving)) == []

    events = supervisor.decide(system(r0_waiting, r1_on_new_cell, r2_done))
    assert Enter(0, (2, 1)) in events
    assert Enter(1, (2, 1)) not in events


# whole runs


def whole_run(layout, seed, orders, capacity, b_low):
    """Random order of uncontrollable events, the supervisor reacts after each of them."""
    model = make_model(layout, capacity=capacity)
    rng = random.Random(seed)
    batch = [
        Order(k, rng.choice(layout.pickups), rng.choice(layout.deliveries), 0.0)
        for k in range(orders)
    ]
    s = initial_state(layout, batch=batch, capacity=capacity)
    supervisor = Supervisor(model, BaselinePolicy(b_low=b_low, seed=seed))
    log = []
    for _ in range(20000):
        events = supervisor.decide(s)
        s = assert_valid_sequence(s, events, model)
        log += events
        assert is_safe_state(s, model)
        waiting = [e for e in enabled_events(s, model) if type(e) in UNCONTROLLABLE]
        if not waiting:
            return s, log, model
        event = rng.choice(waiting)
        s = next_state(s, event, model)
        log.append(event)
        held = [cell for r in s.robots for cell in r.held]
        assert len(held) == len(set(held))
        for i, r in enumerate(s.robots):
            assert r.battery >= nearest_charging_dist(model, pos(s, i))
    pytest.fail("the run did not end")


RUNS = [
    ("mini.yaml", 0, 8, 400, 100),
    ("mini.yaml", 1, 8, 400, 100),
    ("mini.yaml", 2, 8, 400, 100),
    ("mini.yaml", 3, 12, 60, 40),
    ("mini.yaml", 4, 12, 60, 40),
    ("warehouse_15x8.yaml", 0, 20, 120, 90),
]


@pytest.mark.parametrize(("name", "seed", "orders", "capacity", "b_low"), RUNS)
def test_baseline_run_ends_in_the_final_state(name, seed, orders, capacity, b_low):
    layout = load_layout(LAYOUTS / name)
    s, _, _ = whole_run(layout, seed, orders, capacity, b_low)
    home = {*layout.chargers, *layout.parkings}
    assert s.delivered == orders
    assert s.queue == ()
    for r in s.robots:
        assert r.mode is Mode.IDLE
        assert r.order is None
        assert r.goal is None
        assert r.held[0] in home


def test_baseline_really_charges_on_a_small_battery(mini):
    _, log, _ = whole_run(mini, 0, 12, 60, 40)
    assert any(isinstance(e, Charged) for e in log)


def test_baseline_same_seed_gives_the_same_run(mini):
    assert whole_run(mini, 7, 8, 400, 100)[1] == whole_run(mini, 7, 8, 400, 100)[1]


def test_baseline_seed_changes_the_run(mini):
    assert whole_run(mini, 7, 8, 400, 100)[1] != whole_run(mini, 8, 8, 400, 100)[1]


# architecture rule


def test_supervisor_does_not_use_time_graphics_or_ros():
    tree = ast.parse(Path(fleet.supervisor.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        if isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert not imported & {"time", "datetime", "asyncio", "matplotlib", "pygame", "rclpy", "sim"}
