"""Directed lane graph G = (V, A) and fixed shortest routes to buffer cells."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from fleet.layout import Cell, Layout

Graph = dict[Cell, tuple[Cell, ...]]  # cell -> cells reachable in one move


def build_graph(layout: Layout) -> Graph:
    """Edges from lane exits plus buffer <-> access lane cell in both directions."""
    succ = {cell: [d.step(cell) for d in dirs] for cell, dirs in layout.exits.items()}
    for buffer, lane in layout.access.items():
        succ[lane].append(buffer)
        succ[buffer] = [lane]
    return {cell: tuple(nxt) for cell, nxt in succ.items()}


def reverse_graph(graph: Graph) -> Graph:
    """Same cells with every edge reversed (to search backwards from a goal)."""
    pred = {cell: [] for cell in graph}
    for u, succ in graph.items():
        for v in succ:
            pred[v].append(u)
    return {cell: tuple(nxt) for cell, nxt in pred.items()}


def _routes_to_goal(
    layout: Layout, reverse: Graph, goal: Cell
) -> tuple[dict[Cell, Cell], dict[Cell, int]]:
    """Compute the next cell and the distance to `goal` for every cell that can reach it.

    Runs BFS backwards from `goal` along the reversed edges.
    Buffer cells other than `goal` are never expanded, so routes never pass through them.
    """
    next_cell = {}
    distance = {}
    distance[goal] = 0
    queue = deque([goal])

    while queue:
        cell = queue.popleft()
        for pred in reverse[cell]:
            if pred in distance:
                continue
            next_cell[pred] = cell
            distance[pred] = distance[cell] + 1
            if pred not in layout.buffer_cells:
                queue.append(pred)

    return next_cell, distance


@dataclass(frozen=True)
class Routes:
    graph: Graph
    next_towards: dict[Cell, dict[Cell, Cell]]  # goal -> (cell -> next cell on its route)
    distance: dict[Cell, dict[Cell, int]]  # goal -> (cell -> number of moves)

    def next_cell(self, u: Cell, g: Cell) -> Cell:
        """Second cell of route(u, g): where a robot at u heading to g moves next."""
        return self.next_towards[g][u]

    def route(self, u: Cell, g: Cell) -> tuple[Cell, ...]:
        """Fixed shortest path u -> g, both ends included."""
        r = [u]
        next_cell = u
        while next_cell != g:
            next_cell = self.next_cell(next_cell, g)
            r.append(next_cell)

        return tuple(r)

    def delta(self, u: Cell, g: Cell) -> int:
        """δ(u, g): length of route(u, g) in moves."""
        return self.distance[g][u]


def build_routes(layout: Layout) -> Routes:
    """Graph plus one reverse BFS per buffer cell; everything precomputed."""
    graph = build_graph(layout)
    reverse = reverse_graph(graph)
    next_towards, distance = {}, {}
    for buff_cell in layout.buffer_cells:
        buff_next, buff_dis = _routes_to_goal(layout, reverse, buff_cell)
        next_towards[buff_cell] = buff_next
        distance[buff_cell] = buff_dis
    routes = Routes(graph, next_towards, distance)
    return routes
