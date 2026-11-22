"""Order generators for the two scenarios of the proposal.

Continuous mode: orders arrive as a Poisson process with rate lam (orders per second), so the
times between arrivals are exponential with mean 1 / lam. Batch mode: N orders, all at t = 0.

In both modes the pick-up and the delivery station of each order are drawn uniformly from the
layout's pick-up cells P and delivery cells D. Order ids are 0, 1, 2, ... in arrival order.

plot_orders shows the generated orders: the inter-arrival times against the exponential
density lam * exp(-lam * t) and the orders per minute (continuous mode only), and the number
of orders per pick-up and per delivery station against the expected count of a uniform choice.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from random import Random

import matplotlib.pyplot as plt
import numpy as np

from fleet.layout import Cell, Layout
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


def station_bars(ax: plt.Axes, layout: Layout, cells: list[Cell], used: list[Cell], title: str):
    counts = Counter(used)
    ax.bar([layout.buffer_name[c] for c in cells], [counts[c] for c in cells], color="#6ea0e6")
    ax.axhline(len(used) / len(cells), color="black", ls="--", lw=1, label="uniform")
    ax.set_title(title)
    ax.set_ylabel("orders")
    ax.legend()


def plot_orders(
    layout: Layout, orders: list[Order], rate: float | None, title: str, save: Path | None = None
):
    """Show the plot in a window, or write it to save (.png, .pdf, ...); rate None = batch.

    Continuous mode adds a top row: the inter-arrival times against Exp(lambda) and the
    number of orders per minute against lambda * 60.
    """
    continuous = rate is not None
    fig, axes = plt.subplots(2 if continuous else 1, 2, figsize=(11, 7 if continuous else 3.5))
    axes = np.atleast_2d(axes)

    if continuous:
        times = np.array([o.created_at for o in orders])
        gaps = np.diff(times, prepend=0.0)
        ax = axes[0, 0]
        ax.hist(gaps, bins=30, density=True, color="#78be78", label="generated")
        t = np.linspace(0, gaps.max(), 200)
        ax.plot(t, rate * np.exp(-rate * t), "k-", label=f"Exp(lambda = {rate:g})")
        ax.set_title("inter-arrival times")
        ax.set_xlabel("time between orders [s]")
        ax.set_ylabel("density")
        ax.legend()

        ax = axes[0, 1]
        ax.hist(times, bins=np.arange(0, times.max() + 60, 60), color="#f0a050")
        ax.axhline(rate * 60, color="black", ls="--", lw=1, label="lambda * 60 s")
        ax.set_title("orders per minute")
        ax.set_xlabel("arrival time [s]")
        ax.set_ylabel("orders")
        ax.legend()

    row = axes[-1]
    station_bars(row[0], layout, layout.pickups, [o.pickup for o in orders], "pick-up stations")
    station_bars(
        row[1], layout, layout.deliveries, [o.delivery for o in orders], "delivery stations"
    )
    fig.suptitle(f"{title}: {len(orders)} generated orders")
    fig.tight_layout()
    if save is None:
        plt.show()
    else:
        fig.savefig(save)
