import dataclasses
from collections import Counter
from pathlib import Path

import pytest
import yaml

from fleet.events import NewOrder
from sim.scenario import build_model, check_energy, load_scenario, parse_scenario, run_scenario
from sim.simulator import Durations

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "scenarios" / "default.yaml"


@pytest.fixture
def data():
    with open(DEFAULT, encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse(data):
    return parse_scenario(data, base_dir=DEFAULT.parent)


# --- reading a scenario ---


def test_load_default():
    sc = load_scenario(DEFAULT)
    assert sc.name == "default"
    assert sc.layout == (ROOT / "layouts" / "warehouse_15x8.yaml").resolve()
    assert sc.mode == "continuous"
    assert sc.rate == 0.02 and sc.duration == 3600.0
    assert sc.orders is None
    assert (sc.capacity, sc.reserve, sc.queue_max) == (400, 3, 20)
    assert (sc.b_low, sc.seed) == (100, 1)
    assert sc.durations == Durations()


def test_batch_mode(data):
    data["mode"] = "batch"
    sc = parse(data)
    assert sc.orders == 30
    assert sc.rate is None and sc.duration is None


def test_unused_block_may_be_missing(data):
    del data["batch"]
    assert parse(data).mode == "continuous"


def test_durations_optional_and_partial(data):
    del data["durations"]
    assert parse(data).durations == Durations()
    data["durations"] = {"move": [2, 3]}
    assert parse(data).durations == Durations(move=(2, 3))


def test_seed_defaults_to_zero(data):
    del data["seed"]
    assert parse(data).seed == 0


def test_absolute_layout_path(data):
    data["layout"] = str(ROOT / "layouts" / "mini.yaml")
    assert parse(data).layout.name == "mini.yaml"


def test_unknown_mode(data):
    data["mode"] = "Batch"
    with pytest.raises(ValueError, match="mode must be continuous or batch"):
        parse(data)


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
        rate=None,
        duration=None,
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
