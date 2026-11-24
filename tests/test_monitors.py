"""Tests of the runtime monitors S1-S3 and of the switch use_safe (issue #11)."""

import random
from pathlib import Path

import pytest

from fleet.events import Arrive, Enter
from fleet.guards import Model
from fleet.layout import load_layout
from fleet.routes import build_routes
from fleet.state import Mode, Order, RobotState, SystemState, initial_state
from fleet.supervisor import BaselinePolicy, Supervisor
from fleet.transitions import next_state
from sim.simulator import Simulator
from verification.monitors import (
    Monitor,
    MonitorViolation,
    check_all,
    check_collision,
    check_energy,
    find_cycles,
    find_deadlock,
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
TO_P1 = Order(id=1, pickup=P1, delivery=D1, created_at=0.0)
TO_P2 = Order(id=2, pickup=P2, delivery=D1, created_at=0.0)
TO_D1 = Order(id=3, pickup=P1, delivery=D1, created_at=0.0)


def robot(*held, goal=None, mode=Mode.IDLE, battery=100, order=None):
    return RobotState(held=held, mode=mode, battery=battery, order=order, goal=goal)


def system(*robots):
    return SystemState(robots=robots)


def make_model(layout):
    return Model(layout=layout, routes=build_routes(layout), q_max=20, b_res=3, capacity=400)


@pytest.fixture(scope="module")
def model():
    return make_model(load_layout(LAYOUTS / "mini.yaml"))


@pytest.fixture(scope="module")
def main_model():
    return make_model(load_layout(LAYOUTS / "warehouse_15x8.yaml"))


def almost_closed_loop():
    """Three robots on the 4-cell loop of mini, robot 0 can close it by entering (2, 1).

    (2, 2) -> (2, 1) -> (3, 1) -> (3, 2) -> (2, 2)
    Robot 0 has the lowest number, so the baseline policy lets it move first.
    """
    return system(
        robot((1, 1), order=TO_D1, goal=D1, mode=Mode.TO_DROP),
        robot((2, 2), goal=C2, mode=Mode.TO_CHARGE),
        robot((3, 1), order=TO_P2, goal=P2, mode=Mode.TO_PICK),
        robot((3, 2), order=TO_P1, goal=P1, mode=Mode.TO_PICK),
    )


def closed_loop(model):
    """Robot 0 entered (2, 1): every robot waits for the next one."""
    s = next_state(almost_closed_loop(), Enter(0, (2, 1)), model)
    return next_state(s, Arrive(0), model)


# no false alarms


@pytest.mark.parametrize("name", ["warehouse_15x8.yaml", "mini.yaml"])
def test_s0_has_no_violation(name):
    layout = load_layout(LAYOUTS / name)
    assert check_all(initial_state(layout), make_model(layout)) == []


def test_almost_closed_loop_has_no_violation(model):
    assert check_all(almost_closed_loop(), model) == []


# S1


def test_two_robots_in_one_cell_is_a_collision():
    s = system(robot((1, 1)), robot((1, 1)))
    violations = check_collision(s)
    assert [v.prop for v in violations] == ["S1"]
    assert violations[0].robots == (0, 1)


def test_moving_robot_shares_the_target_cell_with_another_robot():
    s = system(robot((1, 1), (2, 1)), robot((2, 1)))
    assert [v.robots for v in check_collision(s)] == [(0, 1)]


def test_neighbouring_robots_are_no_collision():
    s = system(robot((1, 1), (2, 1)), robot((2, 2)))
    assert check_collision(s) == []


# S2


def test_battery_too_low_to_reach_a_charger(model):
    need = min(model.routes.delta(K1, c) for c in model.layout.chargers)
    s = system(robot(K1, battery=need - 1))
    violations = check_energy(s, model)
    assert [v.prop for v in violations] == ["S2"]
    assert violations[0].robots == (0,)


def test_battery_exactly_enough_is_fine(model):
    need = min(model.routes.delta(K1, c) for c in model.layout.chargers)
    assert check_energy(system(robot(K1, battery=need)), model) == []


def test_empty_battery_on_a_charger_is_fine(model):
    assert check_energy(system(robot(C1, battery=0)), model) == []


# S3


def test_cycle_of_three():
    assert find_cycles({0: 1, 1: 2, 2: 0}) == [(0, 1, 2)]


def test_chain_without_cycle():
    assert find_cycles({0: 1, 1: 2}) == []


def test_robot_waiting_for_a_cycle_is_not_part_of_it():
    assert find_cycles({2: 0, 0: 1, 1: 0}) == [(0, 1)]


def test_closed_loop_is_a_deadlock(model):
    violations = find_deadlock(closed_loop(model), model)
    assert [v.prop for v in violations] == ["S3"]
    assert set(violations[0].robots) == {0, 1, 2, 3}


def test_robot_still_moving_does_not_close_the_loop(model):
    # Robot 0 holds (1, 1) and (2, 1) while it moves: it waits for nobody yet.
    s = next_state(almost_closed_loop(), Enter(0, (2, 1)), model)
    assert find_deadlock(s, model) == []


# Monitor


def test_monitor_counts_checked_states(model):
    monitor = Monitor(model)
    monitor.check(0.0, almost_closed_loop())
    monitor.check(1.0, almost_closed_loop())
    assert monitor.checked == 2


def test_monitor_stops_with_a_report(model):
    monitor = Monitor(model)
    with pytest.raises(MonitorViolation) as info:
        monitor.check(12.5, closed_loop(model))
    assert info.value.time == 12.5
    assert "S3" in str(info.value)
    assert "t = 12.500" in str(info.value)


# switch use_safe and sanity check in the simulator


def test_without_safe_the_closing_move_is_a_candidate(model):
    supervisor = Supervisor(model, BaselinePolicy(b_low=100), use_safe=False)
    assert Enter(0, (2, 1)) in supervisor.candidates(almost_closed_loop())


def test_with_safe_the_closing_move_is_not_a_candidate(model):
    supervisor = Supervisor(model, BaselinePolicy(b_low=100))
    assert Enter(0, (2, 1)) not in supervisor.candidates(almost_closed_loop())


def test_sanity_check_without_safe_the_detector_reports_a_deadlock(model):
    supervisor = Supervisor(model, BaselinePolicy(b_low=100), use_safe=False)
    sim = Simulator(model, supervisor, monitor=Monitor(model))
    with pytest.raises(MonitorViolation) as info:
        sim.run(almost_closed_loop(), [])
    assert [v.prop for v in info.value.violations] == ["S3"]
    assert len(info.value.last_events) > 0


def test_with_safe_the_same_scenario_has_no_violation(model):
    supervisor = Supervisor(model, BaselinePolicy(b_low=100))
    monitor = Monitor(model)
    result = Simulator(model, supervisor, monitor=monitor).run(almost_closed_loop(), [])
    assert monitor.checked > 0
    assert result.state.delivered == 3


# Done when: random runs with 10 robots and safe() on, zero violations


def random_orders(layout, count, seed, horizon=600.0):
    rng = random.Random(seed)
    orders = [
        Order(k, rng.choice(layout.pickups), rng.choice(layout.deliveries), rng.uniform(0, horizon))
        for k in range(count)
    ]
    return sorted(orders, key=lambda o: o.created_at)


@pytest.mark.parametrize("seed", range(3))
def test_random_runs_with_ten_robots_have_no_violation(main_model, seed):
    layout = main_model.layout
    supervisor = Supervisor(main_model, BaselinePolicy(b_low=100, seed=seed))
    monitor = Monitor(main_model)
    sim = Simulator(main_model, supervisor, seed=seed, monitor=monitor)
    result = sim.run(initial_state(layout), random_orders(layout, 20, seed))
    assert len(result.state.robots) == 10
    assert result.finished
    assert monitor.checked == len(result.log)
