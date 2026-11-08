import csv
import random
from pathlib import Path

import pytest

from fleet.events import (
    CONTROLLABLE,
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
from fleet.guards import Model
from fleet.layout import load_layout
from fleet.routes import build_routes
from fleet.safety import is_safe_state
from fleet.state import Mode, Order, RobotState, SystemState, initial_state
from fleet.supervisor import BaselinePolicy, Supervisor
from fleet.transitions import next_state
from sim.simulator import (
    Durations,
    LogEntry,
    Result,
    Simulator,
    completion,
    is_final,
    log_entry,
    write_csv,
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
FIXED = Durations(
    move=(2.0, 2.0), loading=(10.0, 10.0), unloading=(5.0, 5.0), charging=(50.0, 50.0)
)


def robot(*held, goal=None, mode=Mode.IDLE, battery=100, order=None):
    return RobotState(held=held, mode=mode, battery=battery, order=order, goal=goal)


def system(*robots, queue=(), delivered=0):
    return SystemState(robots=robots, queue=queue, delivered=delivered)


def make_model(layout, q_max=20, capacity=400):
    return Model(
        layout=layout, routes=build_routes(layout), q_max=q_max, b_res=3, capacity=capacity
    )


@pytest.fixture(scope="module")
def mini():
    return load_layout(LAYOUTS / "mini.yaml")


@pytest.fixture(scope="module")
def main():
    return load_layout(LAYOUTS / "warehouse_15x8.yaml")


@pytest.fixture(scope="module")
def model(mini):
    return make_model(mini)


def simulate(
    layout, orders, durations=FIXED, seed=0, time_limit=None, q_max=20, policy=None, start=None
):
    model = make_model(layout, q_max=q_max)
    policy = policy if policy is not None else BaselinePolicy(b_low=100)
    sim = Simulator(model, Supervisor(model, policy), durations=durations, seed=seed)
    s = start if start is not None else initial_state(layout)
    return sim.run(s, orders, time_limit=time_limit), model


def random_orders(layout, n, seed, horizon):
    rng = random.Random(seed)
    orders = [
        Order(k, rng.choice(layout.pickups), rng.choice(layout.deliveries), rng.uniform(0, horizon))
        for k in range(n)
    ]
    return sorted(orders, key=lambda o: o.created_at)


def replay(layout, model, result):
    """Apply the logged events from s0 one by one; every one of them must be valid."""
    s = initial_state(layout)
    for entry in result.log:
        s = next_state(s, entry.event, model)
        yield entry, s


# durations


def test_default_durations_are_valid_ranges():
    d = Durations()
    for low, high in (d.move, d.loading, d.unloading, d.charging):
        assert 0 < low <= high


# completion


@pytest.mark.parametrize(
    ("started", "ended", "name"),
    [
        (Enter(2, (1, 1)), Arrive(2), "move"),
        (Load(2), Loaded(2), "loading"),
        (Unload(2), Unloaded(2), "unloading"),
        (Charge(2), Charged(2), "charging"),
    ],
    ids=["enter", "load", "unload", "charge"],
)
def test_completion_of_an_action(started, ended, name):
    durations = Durations()
    low, high = getattr(durations, name)
    rng = random.Random(0)
    for _ in range(50):
        event, after = completion(started, durations, rng)
        assert event == ended
        assert low <= after <= high


@pytest.mark.parametrize(
    "event",
    [NewOrder(ORDER), Assign(0, ORDER), Dispatch(0, P1), Arrive(0), Loaded(0), Unloaded(0)],
    ids=lambda e: type(e).__name__,
)
def test_completion_of_events_that_start_no_action(event):
    assert completion(event, Durations(), random.Random(0)) is None


def test_completion_uses_the_given_durations():
    assert completion(Load(0), FIXED, random.Random(0)) == (Loaded(0), 10.0)
    assert completion(Charge(1), FIXED, random.Random(0)) == (Charged(1), 50.0)


def test_completion_draws_different_durations():
    rng = random.Random(0)
    times = {completion(Enter(0, (1, 1)), Durations(), rng)[1] for _ in range(20)}
    assert len(times) > 1


# is_final


def test_s0_without_orders_is_final(model, mini):
    assert is_final(initial_state(mini), 0, model) is True


@pytest.mark.parametrize(
    "s",
    [
        system(robot(K1), robot(C1), delivered=1),
        system(robot(K2), robot(C2), delivered=1),
    ],
)
def test_final_when_everything_is_done(model, s):
    assert is_final(s, 1, model) is True


@pytest.mark.parametrize(
    ("s", "reason"),
    [
        (system(robot(K1), delivered=0), "an order is not delivered"),
        (system(robot(K1), queue=(ORDER,), delivered=1), "the queue is not empty"),
        (system(robot(D1), delivered=1), "a robot stands on a station"),
        (system(robot(K1, goal=C1, mode=Mode.TO_CHARGE), delivered=1), "a robot has a goal"),
        (system(robot(C1, mode=Mode.CHARGING), delivered=1), "a robot is charging"),
        (system(robot(K1, (1, 1), goal=C1, mode=Mode.TO_CHARGE), delivered=1), "moving"),
    ],
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_not_final(model, s, reason):
    assert is_final(s, 1, model) is False, reason


# log entries and CSV


def test_log_entry_of_enter_is_the_cell_entered():
    s = system(robot(K1, (1, 1), goal=C1, mode=Mode.TO_CHARGE))
    assert log_entry(3.0, Enter(0, (1, 1)), s) == LogEntry(3.0, Enter(0, (1, 1)), 0, (1, 1))


def test_log_entry_of_arrive_is_the_new_position():
    s = system(robot((1, 1), goal=C1, mode=Mode.TO_CHARGE))
    assert log_entry(4.0, Arrive(0), s) == LogEntry(4.0, Arrive(0), 0, (1, 1))


def test_log_entry_of_dispatch_is_the_goal():
    s = system(robot(K1, goal=P1, mode=Mode.TO_PICK, order=ORDER))
    assert log_entry(1.0, Dispatch(0, P1), s) == LogEntry(1.0, Dispatch(0, P1), 0, P1)


def test_log_entry_of_new_order_has_no_robot():
    s = system(robot(K1), queue=(ORDER,))
    assert log_entry(0.0, NewOrder(ORDER), s) == LogEntry(0.0, NewOrder(ORDER), None, P1)


def test_log_entry_of_other_events_is_the_robot_position():
    s = system(robot(K1), robot(P1, mode=Mode.LOADING, order=ORDER))
    assert log_entry(9.0, Load(1), s) == LogEntry(9.0, Load(1), 1, P1)


def test_write_csv(tmp_path):
    log = (
        LogEntry(0.0, NewOrder(ORDER), None, P1),
        LogEntry(12.5, Enter(3, (4, 6)), 3, (4, 6)),
        LogEntry(13.25, Arrive(3), 3, (4, 6)),
    )
    path = tmp_path / "log.csv"
    write_csv(log, path)
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    assert rows == [
        ["time", "event", "robot", "cell"],
        ["0.000", "NewOrder", "", "1 3"],
        ["12.500", "Enter", "3", "4 6"],
        ["13.250", "Arrive", "3", "4 6"],
    ]


def test_write_csv_of_a_whole_run(tmp_path, mini):
    result, _ = simulate(mini, [ORDER])
    path = tmp_path / "run.csv"
    write_csv(result.log, path)
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(result.log)
    assert [r["event"] for r in rows] == [type(e.event).__name__ for e in result.log]


# one order from start to end (batch mode)


def test_one_order_is_delivered_and_the_run_ends_in_s_fin(mini, model):
    result, model = simulate(mini, [ORDER])
    assert isinstance(result, Result)
    assert result.finished is True
    assert result.state.delivered == 1
    assert is_final(result.state, 1, model)


def test_one_order_goes_through_all_steps_in_order(mini):
    result, _ = simulate(mini, [ORDER])
    names = [type(e.event).__name__ for e in result.log]
    steps = ["NewOrder", "Assign", "Load", "Loaded", "Unload", "Unloaded"]
    positions = [names.index(step) for step in steps]
    assert positions == sorted(positions)


def test_the_log_replays_to_the_final_state(mini):
    result, model = simulate(mini, [ORDER])
    *_, (_, s) = replay(mini, model, result)
    assert s == result.state


def test_log_cells_match_the_replayed_states(mini):
    result, model = simulate(mini, [ORDER])
    for entry, s in replay(mini, model, result):
        assert entry == log_entry(entry.time, entry.event, s)


def test_times_never_go_back(mini):
    result, _ = simulate(mini, random_orders(mini, 6, seed=1, horizon=300))
    times = [e.time for e in result.log]
    assert times == sorted(times)
    assert result.time == times[-1]


def test_controllable_events_take_no_time(mini):
    # A controllable event happens at the time of the event the supervisor reacted to.
    result, _ = simulate(mini, random_orders(mini, 6, seed=2, horizon=300))
    previous = 0.0
    for entry in result.log:
        if type(entry.event) in CONTROLLABLE:
            assert entry.time == previous
        previous = entry.time


@pytest.mark.parametrize(
    ("started", "ended", "duration"),
    [(Enter, Arrive, 2.0), (Load, Loaded, 10.0), (Unload, Unloaded, 5.0), (Charge, Charged, 50.0)],
    ids=["move", "loading", "unloading", "charging"],
)
def test_actions_take_their_durations(mini, started, ended, duration):
    # Small battery, so that robots also charge during the run.
    model = make_model(mini, capacity=60)
    orders = random_orders(mini, 8, seed=3, horizon=200)
    sim = Simulator(model, Supervisor(model, BaselinePolicy(b_low=40)), durations=FIXED)
    result = sim.run(initial_state(mini, capacity=60), orders)
    start = {}
    seen = 0
    for entry in result.log:
        if isinstance(entry.event, started):
            start[entry.robot] = entry.time
        if isinstance(entry.event, ended):
            assert entry.time == pytest.approx(start.pop(entry.robot) + duration)
            seen += 1
    assert seen > 0


def test_new_orders_appear_at_their_time(mini):
    later = Order(id=2, pickup=P2, delivery=D1, created_at=100.0)
    result, _ = simulate(mini, [ORDER, later])
    new = {e.event.order.id: e.time for e in result.log if isinstance(e.event, NewOrder)}
    assert new == {1: 0.0, 2: 100.0}
    assigned = [e.time for e in result.log if e.event == Assign(e.robot, later)]
    assert len(assigned) == 1
    assert assigned[0] >= 100.0


def test_order_waits_outside_when_the_queue_is_full(mini):
    # 4 robots take 4 orders, the 5th fills the queue (Qmax = 1), the 6th and 7th wait outside.
    orders = [Order(k, P1, D1, 0.0) for k in range(7)]
    result, model = simulate(mini, orders, q_max=1)
    assert result.finished is True
    assert result.state.delivered == 7
    for _, s in replay(mini, model, result):
        assert len(s.queue) <= 1
    new = [e for e in result.log if isinstance(e.event, NewOrder)]
    assert [e.event.order.id for e in new] == list(range(7))  # waiting orders keep FIFO
    assert new[-1].time > 0.0  # the last ones really had to wait


def test_run_can_start_from_any_state(mini, model):
    # No orders at all, but a robot stands on a delivery station: the supervisor must react
    # before the first event, otherwise nothing would ever happen.
    start = system(robot(C1), robot(C2), robot(K2), robot(D1))
    result, _ = simulate(mini, [], start=start)
    assert result.finished is True
    assert result.state.robots[3].held[0] == K1


def test_empty_batch_ends_at_once(mini):
    result, _ = simulate(mini, [])
    assert result.finished is True
    assert result.log == ()
    assert result.time == 0.0


class NothingPolicy:
    def choose(self, s, candidates, model):
        return None


def test_run_stops_when_nothing_more_can_happen(mini):
    # Nobody takes the order, so the event queue runs empty: the run must end, not hang.
    result, _ = simulate(mini, [ORDER], policy=NothingPolicy())
    assert result.finished is False
    assert result.state.queue == (ORDER,)
    assert [type(e.event) for e in result.log] == [NewOrder]


def test_batch_run_stops_as_soon_as_s_fin_is_reached(mini):
    result, model = simulate(mini, [ORDER])
    states = [s for _, s in replay(mini, model, result)]
    assert is_final(states[-1], 1, model)
    assert not any(is_final(s, 1, model) for s in states[:-1])


# continuous mode (time limit)


def test_time_limit_stops_the_run(mini):
    orders = random_orders(mini, 40, seed=4, horizon=2000)
    result, _ = simulate(mini, orders, time_limit=600.0)
    assert result.time == 600.0
    assert all(e.time <= 600.0 for e in result.log)
    assert max(e.time for e in result.log) > 500.0
    assert result.state.delivered > 0


def test_orders_after_the_time_limit_never_appear(mini):
    orders = [ORDER, Order(id=2, pickup=P2, delivery=D1, created_at=900.0)]
    result, _ = simulate(mini, orders, time_limit=600.0)
    ids = [e.event.order.id for e in result.log if isinstance(e.event, NewOrder)]
    assert ids == [1]
    # Order 1 is delivered and the robots are home, but order 2 never came: not s_fin.
    assert result.finished is False


def test_time_limit_mode_does_not_stop_at_s_fin(mini):
    # s_fin is reached early, but the run goes on until the limit: the next order may come.
    later = Order(id=2, pickup=P2, delivery=D1, created_at=500.0)
    result, _ = simulate(mini, [ORDER, later], time_limit=800.0)
    assert result.state.delivered == 2
    assert result.time == 800.0


# seeds


def test_same_seed_gives_the_same_run(mini):
    orders = random_orders(mini, 6, seed=5, horizon=300)
    a, _ = simulate(mini, orders, durations=Durations(), seed=7)
    b, _ = simulate(mini, orders, durations=Durations(), seed=7)
    assert a.log == b.log


def test_seed_changes_the_durations(mini):
    # Same policy, only the seed of the simulator differs.
    orders = random_orders(mini, 6, seed=5, horizon=300)
    a, _ = simulate(mini, orders, durations=Durations(), seed=7)
    b, _ = simulate(mini, orders, durations=Durations(), seed=8)
    assert [e.time for e in a.log] != [e.time for e in b.log]


def test_events_with_equal_times_keep_the_order_of_their_start(mini):
    # With fixed durations, robots that start moving at the same time also arrive at the same
    # time. Their Arrive events must come in the order of their Enter events.
    result, _ = simulate(mini, [Order(k, P1, D1, 0.0) for k in range(4)])
    entered = {}
    pairs = 0
    for entry in result.log:
        if isinstance(entry.event, Enter):
            entered.setdefault(entry.time, []).append(entry.robot)
    arrived = {}
    for entry in result.log:
        if isinstance(entry.event, Arrive):
            arrived.setdefault(entry.time - FIXED.move[0], []).append(entry.robot)
    for time, robots in entered.items():
        if len(robots) > 1:
            assert arrived[time] == robots
            pairs += 1
    assert pairs > 0


# whole runs


@pytest.mark.parametrize(("name", "orders"), [("mini.yaml", 12), ("warehouse_15x8.yaml", 25)])
def test_batch_runs_deliver_everything_and_stay_safe(name, orders):
    layout = load_layout(LAYOUTS / name)
    result, model = simulate(
        layout, random_orders(layout, orders, seed=6, horizon=0), durations=Durations(), seed=6
    )
    assert result.finished is True
    assert result.state.delivered == orders
    for _, s in replay(layout, model, result):
        held = [cell for r in s.robots for cell in r.held]
        assert len(held) == len(set(held))
        assert is_safe_state(s, model)
