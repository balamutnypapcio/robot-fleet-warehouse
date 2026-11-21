"""Order generators for the two scenarios of the proposal.

Continuous mode: orders arrive as a Poisson process with rate lam (orders per second), so the
times between arrivals are exponential with mean 1 / lam. Batch mode: N orders, all at t = 0.

In both modes the pick-up and the delivery station of each order are drawn uniformly from the
layout's pick-up cells P and delivery cells D. Order ids are 0, 1, 2, ... in arrival order.
"""

from __future__ import annotations

from random import Random

from fleet.layout import Layout
from fleet.state import Order


def random_order(layout: Layout, order_id: int, created_at: float, rng: Random) -> Order:
    """One order with a random pick-up and a random delivery station."""
    return Order(
        id=order_id,
        pickup=rng.choice(layout.pickups),
        delivery=rng.choice(layout.deliveries),
        created_at=created_at,
    )


def poisson_orders(layout: Layout, rate: float, horizon: float, rng: Random) -> list[Order]:
    """Orders of a Poisson process with the given rate on [0, horizon], sorted by time."""
    orders: list[Order] = []
    t = rng.expovariate(rate)
    while t <= horizon:
        orders.append(random_order(layout, len(orders), t, rng))
        t += rng.expovariate(rate)
    return orders


def batch_orders(layout: Layout, n: int, rng: Random) -> list[Order]:
    """N orders, all created at t = 0."""
    return [random_order(layout, k, 0.0, rng) for k in range(n)]
