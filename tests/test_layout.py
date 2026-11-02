from pathlib import Path

import pytest
import yaml

from fleet.layout import CellKind, Direction, load_layout, parse_layout, render

LAYOUTS = Path(__file__).resolve().parents[1] / "layouts"


@pytest.fixture(scope="module")
def main():
    return load_layout(LAYOUTS / "warehouse_15x8.yaml")


@pytest.fixture(scope="module")
def mini():
    return load_layout(LAYOUTS / "mini.yaml")


def test_main_buffer_counts(main):
    assert len(main.pickups) == 5
    assert len(main.deliveries) == 3
    assert len(main.chargers) == 4
    assert len(main.parkings) == 6
    assert len(main.chargers) + len(main.parkings) == 10


def test_main_matches_proposal(main):
    assert (main.width, main.height) == (15, 8)
    assert len(main.lanes) == 44
    assert len(main.floor) == 62
    assert main.intersections == {(7, 1), (7, 6), (10, 1), (10, 6)}


def test_main_initial_robot_cells(main):
    # s0: robots 1-4 on chargers along the bottom, robots 5-10 on parking K1 (bottom) .. K6 (top)
    assert main.chargers == ((2, 0), (3, 0), (4, 0), (5, 0))
    assert main.parkings == tuple((0, y) for y in range(1, 7))


@pytest.mark.parametrize("name", ["warehouse_15x8.yaml", "mini.yaml"])
def test_every_buffer_has_exactly_one_lane_neighbour(name):
    layout = load_layout(LAYOUTS / name)
    for cell in layout.buffer_cells:
        lanes = [d.step(cell) for d in Direction if d.step(cell) in layout.lanes]
        assert lanes == [layout.access[cell]]


@pytest.mark.parametrize("name", ["warehouse_15x8.yaml", "mini.yaml"])
def test_every_lane_cell_can_be_entered(name):
    layout = load_layout(LAYOUTS / name)
    entered = {d.step(c) for c, dirs in layout.exits.items() for d in dirs}
    assert entered == layout.lanes


def test_mini(mini):
    assert len(mini.chargers) + len(mini.parkings) == 4
    assert len(mini.pickups) == 2
    assert len(mini.deliveries) == 1
    assert len(mini.lanes) == 6


def test_kind_and_position(main):
    assert main.kind(main.buffers["P3"]) is CellKind.PICKUP
    assert main.kind(main.buffers["D1"]) is CellKind.DELIVERY
    assert main.kind((7, 6)) is CellKind.LANE
    assert main.kind((2, 3)) is CellKind.RACK
    assert main.kind((0, 0)) is CellKind.VOID
    assert main.position((0, 0)) == (0.5, 0.5)


@pytest.mark.parametrize("name", ["warehouse_15x8.yaml", "mini.yaml"])
def test_render_round_trip(name):
    layout = load_layout(LAYOUTS / name)
    with open(LAYOUTS / name, encoding="utf-8") as f:
        grid = yaml.safe_load(f)["grid"]
    assert render(layout) == grid.rstrip()
    assert parse_layout({"grid": render(layout)}).exits == layout.exits


def test_render_robots(mini):
    text = render(mini, robots={mini.buffers["K1"]: "R1"})
    assert text.splitlines()[2].startswith("R1  >")


@pytest.mark.parametrize(
    ("grid", "message"),
    [
        ("K1 > ^\n.  P1", "row 2 has 2 cells"),
        ("K1 > ^\nP1 P1 .", "duplicate buffer"),
        ("K1 > X\n.  P1 .", "unknown grid token"),
        ("K1 > >", "non-lane cell"),
        ("P1 #  .\n.  .  .", "0 neighbouring lane cells"),
        ("v  <  <\nv  P1 ^\n>  >  ^", "4 neighbouring lane cells"),
    ],
)
def test_invalid_layouts(grid, message):
    with pytest.raises(ValueError, match=message):
        parse_layout({"grid": grid})
