import random
import statistics
import time
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
from fleet.guards import Model, enabled_events
from fleet.layout import load_layout
from fleet.routes import build_routes
from fleet.safety import is_safe_state, safe
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
TO_P1 = Order(id=1, pickup=P1, delivery=D1, created_at=0.0)
TO_P2 = Order(id=2, pickup=P2, delivery=D1, created_at=0.0)


def robot(*held, goal=None, mode=Mode.IDLE, battery=100, order=None):
    return RobotState(held=held, mode=mode, battery=battery, order=order, goal=goal)


def system(*robots, queue=()):
    return SystemState(robots=robots, queue=queue)


def make_model(layout, capacity=400):
    return Model(layout=layout, routes=build_routes(layout), q_max=20, b_res=3, capacity=capacity)


@pytest.fixture(scope="module")
def main():
    return load_layout(LAYOUTS / "warehouse_15x8.yaml")


@pytest.fixture(scope="module")
def mini():
    return load_layout(LAYOUTS / "mini.yaml")


@pytest.fixture(scope="module")
def model(mini):
    return make_model(mini)


@pytest.fixture(scope="module")
def main_model(main):
    return make_model(main)


def almost_closed_loop():
    """Three robots on the 4-cell loop of mini; robot 3 can still close it or stay out.

    (2, 2) -> (2, 1) -> (3, 1) -> (3, 2) -> (2, 2)
    Robot 0 can leave first through (2, 1) to C2, which lets robots 2 and 1 leave too.
    """
    return system(
        robot((2, 2), goal=C2, mode=Mode.TO_CHARGE),
        robot((3, 1), order=TO_P2, goal=P2, mode=Mode.TO_PICK),
        robot((3, 2), order=TO_P1, goal=P1, mode=Mode.TO_PICK),
        robot((1, 1), order=TO_P1, goal=D1, mode=Mode.TO_DROP),
    )


# the 4-cell loop (Done when)


def test_almost_closed_loop_is_safe(model):
    assert is_safe_state(almost_closed_loop(), model) is True


def test_move_that_closes_the_loop_is_rejected(model):
    assert safe(almost_closed_loop(), Enter(3, (2, 1)), model) is False


def test_closed_loop_is_not_a_safe_state(model):
    closed = next_state(almost_closed_loop(), Enter(3, (2, 1)), model)
    assert is_safe_state(closed, model) is False


def test_move_of_the_robot_that_opens_the_loop_is_accepted(model):
    assert safe(almost_closed_loop(), Enter(0, (2, 1)), model) is True


# every first move from s0 (Done when)


@pytest.mark.parametrize("name", ["warehouse_15x8.yaml", "mini.yaml"])
def test_every_first_move_from_s0_is_safe(name):
    layout = load_layout(LAYOUTS / name)
    model = make_model(layout)
    for pickup in layout.pickups:
        order = Order(id=1, pickup=pickup, delivery=layout.deliveries[0], created_at=0.0)
        for i in range(len(layout.chargers) + len(layout.parkings)):
            s = initial_state(layout, batch=[order])
            s = next_state(s, Assign(i, order), model)
            s = next_state(s, Dispatch(i, pickup), model)
            assert safe(s, Enter(i, next_cell(s, i, model.routes)), model) is True


@pytest.mark.parametrize("name", ["warehouse_15x8.yaml", "mini.yaml"])
def test_s0_is_a_safe_state(name):
    layout = load_layout(LAYOUTS / name)
    assert is_safe_state(initial_state(layout), make_model(layout)) is True


# events other than enter


@pytest.mark.parametrize(
    "event",
    [
        NewOrder(TO_P1),
        Assign(0, TO_P1),
        Dispatch(0, P1),
        Arrive(0),
        Load(0),
        Loaded(0),
        Unload(0),
        Unloaded(0),
        Charge(0),
        Charged(0),
    ],
    ids=lambda e: type(e).__name__,
)
def test_other_events_are_always_safe(model, event):
    # Algorithm 1, line 1: true without looking at the state, even in an unsafe one.
    closed = next_state(almost_closed_loop(), Enter(3, (2, 1)), model)
    assert safe(closed, event, model) is True
    assert safe(system(robot(K1)), event, model) is True


# which robots are active


def test_no_active_robots_is_safe(model):
    s = system(robot(K1), robot(C1), robot(K2, goal=P1, mode=Mode.TO_PARK))
    assert is_safe_state(s, model) is True


def test_single_robot_is_always_safe(model):
    assert is_safe_state(system(robot((2, 2), goal=P2, mode=Mode.TO_PARK)), model) is True


def test_leaving_a_buffer_cell_can_close_a_cycle(main, main_model):
    # Robot 0 drives past (4, 1) to C3. Robot 1 on (4, 1) must go around the whole loop and
    # through (2, 1) to C2. If robot 0 takes (2, 1) now, each blocks the route of the other.
    c1, c2, c3 = main.buffers["C1"], main.buffers["C2"], main.buffers["C3"]
    s = system(
        robot(c1, goal=c3, mode=Mode.TO_CHARGE),
        robot((4, 1), goal=c2, mode=Mode.TO_CHARGE),
    )
    assert is_safe_state(s, main_model) is True
    assert safe(s, Enter(0, (2, 1)), main_model) is False
    assert safe(s, Enter(1, (5, 1)), main_model) is True


def test_only_the_whole_remaining_route_counts_not_only_the_next_cell(main, main_model):
    # The next cell of both robots is free, but further on their routes block each other.
    c1, c2, c3 = main.buffers["C1"], main.buffers["C2"], main.buffers["C3"]
    s = system(
        robot(c1, (2, 1), goal=c3, mode=Mode.TO_CHARGE),
        robot((4, 1), goal=c2, mode=Mode.TO_CHARGE),
    )
    assert is_safe_state(s, main_model) is False


def test_robot_arriving_at_its_goal_finishes_and_releases_the_lane_cell(model):
    s = system(
        robot((1, 1), C1, goal=C1, mode=Mode.TO_CHARGE),
        robot((1, 2), goal=C2, mode=Mode.TO_CHARGE),
    )
    assert is_safe_state(s, model) is True


# releasing cells and repeating the search


def test_robot_releases_both_held_cells(model):
    # Robot 1 needs both cells of the moving robot 0: (1, 1) and (2, 1).
    s = system(
        robot((1, 1), (2, 1), goal=C2, mode=Mode.TO_CHARGE),
        robot((1, 2), order=TO_P1, goal=D1, mode=Mode.TO_DROP),
    )
    assert is_safe_state(s, model) is True


QUEUE = [
    (robot((3, 6), goal=(0, 4), mode=Mode.TO_PARK), "head"),
    (robot((4, 6), goal=(0, 5), mode=Mode.TO_PARK), "middle"),
    (robot((5, 6), goal=(0, 6), mode=Mode.TO_PARK), "tail"),
]


@pytest.mark.parametrize("order", [(0, 1, 2), (2, 1, 0), (1, 2, 0), (2, 0, 1)])
def test_a_queue_finishes_whatever_the_robot_numbering(main_model, order):
    # Only the head can start, so a single pass over the robots may not be enough.
    s = system(*(QUEUE[k][0] for k in order))
    assert is_safe_state(s, main_model) is True


def test_a_later_robot_can_unblock_an_earlier_one(model):
    # Robot 0 waits for (1, 1) held by robot 1, which can leave at once.
    s = system(
        robot((1, 2), goal=C2, mode=Mode.TO_CHARGE),
        robot((1, 1), goal=C1, mode=Mode.TO_CHARGE),
    )
    assert is_safe_state(s, model) is True


# runtime (Done when)


def test_runtime_below_one_millisecond_for_ten_robots(main):
    # The worst order: only the last robot can start, so every pass removes one robot.
    model = make_model(main)
    goals = [main.buffers[n] for n in ["K1", "K2", "K3", "K4", "K5", "K6", "C1", "C2", "C3", "C4"]]
    robots = [
        robot((x, 6), goal=goal, mode=Mode.TO_PARK if goal[0] == 0 else Mode.TO_CHARGE)
        for x, goal in zip(range(12, 2, -1), goals, strict=True)
    ]
    s = system(*robots)
    e = Enter(9, (2, 6))
    assert safe(s, e, model) is True

    times = []
    for _ in range(101):
        start = time.perf_counter()
        safe(s, e, model)
        times.append(time.perf_counter() - start)
    assert statistics.median(times) < 0.001


# random runs


def circular_wait(s, model):
    """True if some robots wait in a cycle, each for a cell held by the next one (S3)."""
    holder = {cell: i for i, r in enumerate(s.robots) for cell in r.held}
    for start in range(len(s.robots)):
        seen = set()
        i = start
        while i is not None and i not in seen:
            seen.add(i)
            r = s.robots[i]
            step = next_cell(s, i, model.routes) if r.goal is not None else None
            i = holder.get(step) if len(r.held) == 1 else None
            if i == start:
                return True
    return False


def random_run(layout, model, seed, steps, use_safe):
    """Yield the states of a run of random enabled events, filtered by safe() if requested."""
    rng = random.Random(seed)
    batch = [
        Order(k, rng.choice(layout.pickups), rng.choice(layout.deliveries), 0.0) for k in range(8)
    ]
    s = initial_state(layout, batch=batch, capacity=model.capacity)
    yield s
    for _ in range(steps):
        events = enabled_events(s, model)
        if use_safe:
            events = [e for e in events if safe(s, e, model)]
        if not events:
            return
        s = next_state(s, rng.choice(events), model)
        yield s


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("name", ["warehouse_15x8.yaml", "mini.yaml"])
def test_supervised_runs_stay_safe(name, seed):
    # Sec. 2.6: every state reachable under the supervisor is safe, so no circular wait exists.
    layout = load_layout(LAYOUTS / name)
    model = make_model(layout)
    for s in random_run(layout, model, seed, steps=400, use_safe=True):
        assert is_safe_state(s, model) is True, f"reached an unsafe state: {s}"
        assert not circular_wait(s, model), f"circular wait: {s}"


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("name", ["warehouse_15x8.yaml", "mini.yaml"])
def test_supervised_runs_never_get_stuck_on_a_lane(name, seed):
    layout = load_layout(LAYOUTS / name)
    model = make_model(layout)
    *_, s = random_run(layout, model, seed, steps=400, use_safe=True)
    if not [e for e in enabled_events(s, model) if safe(s, e, model)]:
        on_lanes = [r for r in s.robots if set(r.held) & layout.lanes]
        assert on_lanes == [], f"robots stuck on lanes: {on_lanes}"


def test_supervised_runs_really_deliver_orders(main):
    # Guards against a supervisor that is safe only because it rejects every move.
    model = make_model(main)
    delivered = 0
    for seed in range(3):
        *_, s = random_run(main, model, seed, steps=400, use_safe=True)
        delivered += s.delivered
    assert delivered > 0


def test_guards_alone_allow_the_move_that_closes_the_loop(model):
    # Shows that safe() is needed at all: this move is in EF(s), but it ends in a circular wait.
    s = almost_closed_loop()
    assert Enter(3, (2, 1)) in enabled_events(s, model)
    assert not circular_wait(s, model)
    s = next_state(s, Enter(3, (2, 1)), model)
    s = next_state(s, Arrive(3), model)  # the robot stands on (2, 1): nobody can move now
    assert circular_wait(s, model)
    assert [e for e in enabled_events(s, model) if isinstance(e, Enter)] == []


# station deadlock (issue #32): safe() only checks enter, so the guard of assign must prevent it


def test_robot_gets_a_new_order_only_after_driving_home(model):
    s = system(robot(D1, mode=Mode.UNLOADING, order=TO_P1), queue=(TO_P2,))
    s = next_state(s, Unloaded(0), model)
    assert Assign(0, TO_P2) not in enabled_events(s, model)

    s = next_state(s, Dispatch(0, K1), model)
    while s.robots[0].goal is not None:
        assert Assign(0, TO_P2) not in enabled_events(s, model)
        s = next_state(s, Enter(0, next_cell(s, 0, model.routes)), model)
        s = next_state(s, Arrive(0), model)

    assert s.robots[0].held == (K1,)
    assert s.robots[0].mode is Mode.IDLE
    assert Assign(0, TO_P2) in enabled_events(s, model)


@pytest.mark.parametrize("seed", range(20))
def test_supervised_runs_on_mini_deliver_the_whole_batch(model, seed):
    for s in random_run(model.layout, model, seed, steps=6000, use_safe=True):
        if s.delivered == 8:  # the size of the batch in random_run
            return
    pytest.fail(f"delivered only {s.delivered} of 8 orders, last state: {s}")
