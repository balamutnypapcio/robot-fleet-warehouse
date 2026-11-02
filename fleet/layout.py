"""Warehouse layout: grid cells, lane directions and buffer cells, loaded from a YAML file.

Grid format (one whitespace-separated token per cell, first line = top row):

    .            void (not part of the floor)
    #            storage rack
    < > ^ v      lane cell; each arrow is an admissible exit direction (one or two arrows)
    +<  +<v ...  intersection: a lane cell where routes merge or split
    P1 D1 C1 K1  buffer cells: pick-up, delivery, charger, parking (any positive number)

Cells are (x, y) tuples of ints with the origin in the bottom-left corner, x to the right
and y up (the same axes as the Gazebo world).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from pathlib import Path

import yaml

Cell = tuple[int, int]


class Direction(Enum):
    N = (0, 1)
    E = (1, 0)
    S = (0, -1)
    W = (-1, 0)

    def step(self, cell: Cell) -> Cell:
        return (cell[0] + self.value[0], cell[1] + self.value[1])


class CellKind(Enum):
    VOID = "void"
    RACK = "rack"
    LANE = "lane"
    PICKUP = "pickup"
    DELIVERY = "delivery"
    CHARGER = "charger"
    PARKING = "parking"


ARROWS = {"^": Direction.N, ">": Direction.E, "v": Direction.S, "<": Direction.W}
BUFFER_KINDS = {
    "P": CellKind.PICKUP,
    "D": CellKind.DELIVERY,
    "C": CellKind.CHARGER,
    "K": CellKind.PARKING,
}

_BUFFER_RE = re.compile(r"([PDCK])([1-9][0-9]*)")
_LANE_RE = re.compile(r"(\+?)([<>^v]{1,2})")


@dataclass(frozen=True)
class Layout:
    name: str
    width: int
    height: int
    cell_size: float
    exits: dict[Cell, tuple[Direction, ...]]  # lane cell -> admissible exit directions
    intersections: frozenset[Cell]
    racks: frozenset[Cell]
    buffers: dict[str, Cell]  # buffer name ("P1", "C3", ...) -> cell
    access: dict[Cell, Cell]  # buffer cell -> its (only) neighbouring lane cell

    @cached_property
    def lanes(self) -> frozenset[Cell]:
        return frozenset(self.exits)

    @cached_property
    def buffer_cells(self) -> frozenset[Cell]:
        return frozenset(self.buffers.values())

    @cached_property
    def floor(self) -> frozenset[Cell]:
        """V = L ∪ B: all cells a robot can stand on."""
        return self.lanes | self.buffer_cells

    @cached_property
    def buffer_name(self) -> dict[Cell, str]:
        return {cell: name for name, cell in self.buffers.items()}

    def _buffers_of(self, prefix: str) -> tuple[Cell, ...]:
        names = sorted((n for n in self.buffers if n[0] == prefix), key=lambda n: int(n[1:]))
        return tuple(self.buffers[n] for n in names)

    @cached_property
    def pickups(self) -> tuple[Cell, ...]:
        return self._buffers_of("P")

    @cached_property
    def deliveries(self) -> tuple[Cell, ...]:
        return self._buffers_of("D")

    @cached_property
    def chargers(self) -> tuple[Cell, ...]:
        return self._buffers_of("C")

    @cached_property
    def parkings(self) -> tuple[Cell, ...]:
        return self._buffers_of("K")

    def kind(self, cell: Cell) -> CellKind:
        if cell in self.exits:
            return CellKind.LANE
        if cell in self.racks:
            return CellKind.RACK
        name = self.buffer_name.get(cell)
        if name is not None:
            return BUFFER_KINDS[name[0]]
        return CellKind.VOID

    def position(self, cell: Cell) -> tuple[float, float]:
        """Metric coordinates of the cell centre."""
        return ((cell[0] + 0.5) * self.cell_size, (cell[1] + 0.5) * self.cell_size)


def load_layout(path: str | Path) -> Layout:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return parse_layout(data)


def parse_layout(data: dict) -> Layout:
    """Build a Layout from the parsed YAML mapping; raise ValueError on invalid data."""
    rows = [line.split() for line in data["grid"].strip().splitlines()]
    height = len(rows)
    width = len(rows[0])
    for i, row in enumerate(rows):
        if len(row) != width:
            raise ValueError(f"grid row {i + 1} has {len(row)} cells, expected {width}")

    exits: dict[Cell, tuple[Direction, ...]] = {}
    intersections: set[Cell] = set()
    racks: set[Cell] = set()
    buffers: dict[str, Cell] = {}

    for i, row in enumerate(rows):
        y = height - 1 - i
        for x, token in enumerate(row):
            cell = (x, y)
            if token == ".":
                continue
            if token == "#":
                racks.add(cell)
            elif _BUFFER_RE.fullmatch(token):
                if token in buffers:
                    raise ValueError(f"duplicate buffer cell {token}")
                buffers[token] = cell
            elif m := _LANE_RE.fullmatch(token):
                dirs = tuple(ARROWS[a] for a in m.group(2))
                if len(set(dirs)) != len(dirs):
                    raise ValueError(f"repeated arrow in {token!r} at {cell}")
                exits[cell] = dirs
                if m.group(1):
                    intersections.add(cell)
            else:
                raise ValueError(f"unknown grid token {token!r} at {cell}")

    for cell, dirs in exits.items():
        for d in dirs:
            if d.step(cell) not in exits:
                raise ValueError(f"lane cell {cell} points {d.name} to a non-lane cell")

    access: dict[Cell, Cell] = {}
    for name, cell in buffers.items():
        lanes = [n for n in (d.step(cell) for d in Direction) if n in exits]
        if len(lanes) != 1:
            raise ValueError(f"buffer {name} at {cell} has {len(lanes)} neighbouring lane cells")
        access[cell] = lanes[0]

    return Layout(
        name=data.get("name", ""),
        width=width,
        height=height,
        cell_size=float(data.get("cell_size", 1.0)),
        exits=exits,
        intersections=frozenset(intersections),
        racks=frozenset(racks),
        buffers=buffers,
        access=access,
    )


_ARROW_OF = {d: a for a, d in ARROWS.items()}


def render(layout: Layout, robots: dict[Cell, str] | None = None) -> str:
    """ASCII picture of the layout in the grid file format; `robots` overlays labels on cells."""
    robots = robots or {}
    lines = []
    for y in range(layout.height - 1, -1, -1):
        tokens = []
        for x in range(layout.width):
            cell = (x, y)
            if cell in robots:
                token = robots[cell]
            elif cell in layout.exits:
                token = "+" if cell in layout.intersections else ""
                token += "".join(_ARROW_OF[d] for d in layout.exits[cell])
            elif cell in layout.racks:
                token = "#"
            else:
                token = layout.buffer_name.get(cell, ".")
            tokens.append(f"{token:<3}")
        lines.append(" ".join(tokens).rstrip())
    return "\n".join(lines)
