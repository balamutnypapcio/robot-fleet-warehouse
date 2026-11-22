import dataclasses
from collections import Counter
from pathlib import Path

import pytest
import yaml

from fleet.events import NewOrder
from sim.scenario import build_model, check_energy, load_scenario, run_scenario
from sim.simulator import Durations

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "scenarios" / "default.yaml"


def write(tmp_path, **changes):
    """The default scenario with some top-level keys changed, as a file in tmp_path."""
    with open(DEFAULT, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data["layout"] = str(ROOT / "layouts" / "warehouse_15x8.yaml")
    data.update(changes)
    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_load_default():
    sc = load_scenario(DEFAULT)
    assert sc.name == "default"
    assert sc.layout == (ROOT / "layouts" / "warehouse_15x8.yaml").resolve()
    assert sc.mode == "continuous"
    assert sc.rate == 0.02 and sc.duration == 3600.0
    assert sc.orders == 30
    assert (sc.capacity, sc.reserve, sc.queue_max) == (400, 3, 20)
    assert (sc.b_low, sc.seed) == (100, 1)
    assert sc.durations == Durations()


def test_partial_durations(tmp_path):
    sc = load_scenario(write(tmp_path, durations={"move": [2, 3]}))
    assert sc.durations == Durations(move=(2, 3))


def test_unknown_mode(tmp_path):
    with pytest.raises(ValueError, match="mode must be continuous or batch"):
        load_scenario(write(tmp_path, mode="Batch"))


# --- building and running a scenario ---

FAST = Durations(move=(1.0, 1.0), loading=(2.0, 2.0), unloading=(2.0, 2.0), charging=(5.0, 5.0))


@pytest.fixture
def mini():
    """Batch scenario on the small layout with short durations."""
    sc = load_scenario(DEFAULT)
    return dataclasses.replace(
        sc,
        layout=(ROOT / "layouts" / "mini.yaml").resolve(),
        mode="batch",
        orders=3,
        queue_max=2,
        b_low=40,
        durations=FAST,
    )


def test_build_model_uses_scenario_values():
    sc = dataclasses.replace(load_scenario(DEFAULT), capacity=300, reserve=5, queue_max=7)
    model = build_model(sc)
    assert (model.capacity, model.b_res, model.q_max) == (300, 5, 7)
    assert model.layout.name == "warehouse_15x8"


def test_default_scenario_passes_energy_check():
    sc = load_scenario(DEFAULT)
    check_energy(sc, build_model(sc))


@pytest.mark.parametrize("change", [{"b_low": 10}, {"capacity": 20}])
def test_energy_check_fails(mini, change):
    sc = dataclasses.replace(mini, **change)
    with pytest.raises(ValueError, match="largest energy need of an order"):
        check_energy(sc, build_model(sc))


def test_batch_run_reaches_s_fin(mini):
    result, orders = run_scenario(mini)
    assert len(orders) == 3
    assert result.finished
    assert result.state.delivered == 3


def test_batch_with_more_orders_than_queue_max(mini):
    sc = dataclasses.replace(mini, orders=6)  # queue_max = 2
    result, _ = run_scenario(sc)
    assert result.finished
    assert result.state.delivered == 6


def test_continuous_run_stops_at_duration():
    sc = dataclasses.replace(load_scenario(DEFAULT), rate=0.05, duration=600.0)
    result, orders = run_scenario(sc)
    assert result.time == 600.0
    assert all(e.time <= 600.0 for e in result.log)
    new_orders = Counter(type(e.event) for e in result.log)[NewOrder]
    assert 0 < new_orders <= len(orders)


def test_same_seed_same_run(mini):
    a, _ = run_scenario(mini)
    b, _ = run_scenario(mini)
    assert a.log == b.log


def test_other_seed_other_run(mini):
    a, _ = run_scenario(mini)
    b, _ = run_scenario(dataclasses.replace(mini, seed=mini.seed + 10))
    assert a.log != b.log
