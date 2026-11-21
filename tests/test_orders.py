from pathlib import Path
from random import Random

import pytest

from fleet.layout import load_layout
from sim.orders import batch_orders, poisson_orders

LAYOUTS = Path(__file__).resolve().parents[1] / "layouts"


@pytest.fixture(scope="module")
def layout():
    return load_layout(LAYOUTS / "warehouse_15x8.yaml")


def test_batch_orders_all_at_zero(layout):
    orders = batch_orders(layout, 25, Random(0))
    assert len(orders) == 25
    assert [o.id for o in orders] == list(range(25))
    assert all(o.created_at == 0.0 for o in orders)
    assert all(o.pickup in layout.pickups and o.delivery in layout.deliveries for o in orders)


def test_poisson_orders_sorted_within_horizon(layout):
    orders = poisson_orders(layout, rate=0.1, horizon=1000.0, rng=Random(3))
    times = [o.created_at for o in orders]
    assert times == sorted(times)
    assert all(0 < t <= 1000.0 for t in times)
    assert [o.id for o in orders] == list(range(len(orders)))
    assert all(o.pickup in layout.pickups and o.delivery in layout.deliveries for o in orders)


def test_poisson_mean_count(layout):
    # E[count] = rate * horizon = 500, std ~ 22; average over seeds to keep the test stable
    counts = [len(poisson_orders(layout, 0.5, 1000.0, Random(seed))) for seed in range(20)]
    assert abs(sum(counts) / len(counts) - 500) < 20


def test_same_seed_same_orders(layout):
    a = poisson_orders(layout, 0.2, 500.0, Random(7))
    b = poisson_orders(layout, 0.2, 500.0, Random(7))
    assert a == b
