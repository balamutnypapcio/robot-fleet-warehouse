import copy
from pathlib import Path

import pytest

from fleet import routes
from fleet.layout import load_layout

LAYOUTS = Path(__file__).resolve().parents[1] / "layouts"
LAYOUT_FILES = ["warehouse_15x8.yaml", "mini.yaml"]

#   .   P1  #   P2
#   K2  v   +<v <
#   K1  >   +>  ^
#   .   C1  C2  D1
MINI_GRAPH = {
    (1, 3): {(1, 2)},  # P1
    (3, 3): {(3, 2)},  # P2
    (0, 2): {(1, 2)},  # K2
    (0, 1): {(1, 1)},  # K1
    (1, 0): {(1, 1)},  # C1
    (2, 0): {(2, 1)},  # C2
    (3, 0): {(3, 1)},  # D1
    (1, 2): {(1, 1), (1, 3), (0, 2)},
    (2, 2): {(1, 2), (2, 1)},
    (3, 2): {(2, 2), (3, 3)},
    (1, 1): {(2, 1), (0, 1), (1, 0)},
    (2, 1): {(3, 1), (2, 0)},
    (3, 1): {(3, 2), (3, 0)},
}

MINI_PREDECESSORS = {
    (1, 3): {(1, 2)},
    (3, 3): {(3, 2)},
    (0, 2): {(1, 2)},
    (0, 1): {(1, 1)},
    (1, 0): {(1, 1)},
    (2, 0): {(2, 1)},
    (3, 0): {(3, 1)},
    (1, 2): {(2, 2), (1, 3), (0, 2)},
    (2, 2): {(3, 2)},
    (3, 2): {(3, 1), (3, 3)},
    (1, 1): {(1, 2), (0, 1), (1, 0)},
    (2, 1): {(2, 2), (1, 1), (2, 0)},
    (3, 1): {(2, 1), (3, 0)},
}

MINI_DISTANCE_TO_P2 = {
    (3, 3): 0,
    (3, 2): 1,
    (3, 1): 2,
    (2, 1): 3,
    (3, 0): 3,
    (2, 2): 4,
    (1, 1): 4,
    (2, 0): 4,
    (1, 2): 5,
    (0, 1): 5,
    (1, 0): 5,
    (1, 3): 6,
    (0, 2): 6,
}

MINI_NEXT_TO_P2 = {
    (3, 2): (3, 3),
    (3, 1): (3, 2),
    (2, 1): (3, 1),
    (3, 0): (3, 1),
    (2, 2): (2, 1),
    (1, 1): (2, 1),
    (2, 0): (2, 1),
    (1, 2): (1, 1),
    (0, 1): (1, 1),
    (1, 0): (1, 1),
    (1, 3): (1, 2),
    (0, 2): (1, 2),
}


def as_sets(graph):
    return {cell: set(cells) for cell, cells in graph.items()}


def edges(graph):
    return {(u, v) for u, cells in graph.items() for v in cells}


def edge_count(graph):
    return sum(len(cells) for cells in graph.values())


def walk(next_cell, start, goal):
    """Follow the next-cell table from start to goal; fail instead of looping forever."""
    path = [start]
    while path[-1] != goal:
        assert len(path) <= len(next_cell) + 1, f"route from {start} does not reach {goal}"
        path.append(next_cell[path[-1]])
    return path


@pytest.fixture(scope="module", params=LAYOUT_FILES)
def layout(request):
    return load_layout(LAYOUTS / request.param)


@pytest.fixture(scope="module")
def main():
    return load_layout(LAYOUTS / "warehouse_15x8.yaml")


@pytest.fixture(scope="module")
def mini():
    return load_layout(LAYOUTS / "mini.yaml")


# build_graph


def test_build_graph_mini_exact(mini):
    assert as_sets(routes.build_graph(mini)) == MINI_GRAPH


def test_build_graph_contains_every_floor_cell(layout):
    assert set(routes.build_graph(layout)) == layout.floor


def test_build_graph_edges_stay_on_the_floor(layout):
    graph = routes.build_graph(layout)
    for u, v in edges(graph):
        assert v in layout.floor, f"edge {u} -> {v} leaves the floor"


def test_build_graph_contains_every_lane_exit(layout):
    graph = routes.build_graph(layout)
    for cell, directions in layout.exits.items():
        for direction in directions:
            assert direction.step(cell) in graph[cell]


def test_build_graph_buffer_connects_only_to_its_access_cell(layout):
    graph = routes.build_graph(layout)
    for buffer, lane in layout.access.items():
        assert tuple(graph[buffer]) == (lane,)
        assert buffer in graph[lane]


def test_build_graph_edge_counts(main, mini):
    # lane exits + (in and out) * number of buffers
    assert edge_count(routes.build_graph(main)) == 46 + 2 * 18
    assert edge_count(routes.build_graph(mini)) == 7 + 2 * 7


def test_build_graph_values_are_tuples(layout):
    assert all(isinstance(cells, tuple) for cells in routes.build_graph(layout).values())


# reverse_graph


def test_reverse_graph_small_example():
    # (0, 0) -> (1, 0) -> (2, 0), and (1, 1) -> (1, 0)
    graph = {
        (0, 0): ((1, 0),),
        (1, 0): ((2, 0),),
        (2, 0): (),
        (1, 1): ((1, 0),),
    }
    assert as_sets(routes.reverse_graph(graph)) == {
        (0, 0): set(),
        (1, 0): {(0, 0), (1, 1)},
        (2, 0): {(1, 0)},
        (1, 1): set(),
    }


def test_reverse_graph_mini_exact(mini):
    graph = routes.build_graph(mini)
    assert as_sets(routes.reverse_graph(graph)) == MINI_PREDECESSORS


def test_reverse_graph_flips_every_edge(layout):
    graph = routes.build_graph(layout)
    reverse = routes.reverse_graph(graph)
    assert set(reverse) == set(graph)
    assert edges(reverse) == {(v, u) for u, v in edges(graph)}
    assert edge_count(reverse) == edge_count(graph)


def test_reverse_graph_twice_gives_the_original(layout):
    graph = routes.build_graph(layout)
    assert as_sets(routes.reverse_graph(routes.reverse_graph(graph))) == as_sets(graph)


def test_reverse_graph_does_not_change_its_input(layout):
    graph = routes.build_graph(layout)
    before = copy.deepcopy(graph)
    routes.reverse_graph(graph)
    assert graph == before


def test_reverse_graph_values_are_tuples(layout):
    reverse = routes.reverse_graph(routes.build_graph(layout))
    assert all(isinstance(cells, tuple) for cells in reverse.values())


# _routes_to_goal


def solve(layout, goal):
    reverse = routes.reverse_graph(routes.build_graph(layout))
    return routes._routes_to_goal(layout, reverse, goal)


def test_routes_to_goal_mini_distances(mini):
    _, distance = solve(mini, mini.buffers["P2"])
    assert distance == MINI_DISTANCE_TO_P2


def test_routes_to_goal_mini_next_cells(mini):
    next_cell, _ = solve(mini, mini.buffers["P2"])
    assert next_cell == MINI_NEXT_TO_P2


def test_routes_to_goal_mini_full_route(mini):
    goal = mini.buffers["P2"]
    next_cell, _ = solve(mini, goal)
    assert walk(next_cell, mini.buffers["K2"], goal) == [
        (0, 2),
        (1, 2),
        (1, 1),
        (2, 1),
        (3, 1),
        (3, 2),
        (3, 3),
    ]


def test_routes_to_goal_goal_itself(layout):
    for goal in layout.buffer_cells:
        next_cell, distance = solve(layout, goal)
        assert distance[goal] == 0
        assert goal not in next_cell


def test_routes_to_goal_every_cell_reaches_every_buffer(layout):
    for goal in layout.buffer_cells:
        next_cell, distance = solve(layout, goal)
        assert set(distance) == layout.floor
        assert set(next_cell) == layout.floor - {goal}


def test_routes_to_goal_steps_follow_graph_edges(layout):
    graph = routes.build_graph(layout)
    for goal in layout.buffer_cells:
        next_cell, _ = solve(layout, goal)
        for cell, step in next_cell.items():
            assert step in graph[cell], f"{cell} -> {step} is not an edge (goal {goal})"


def test_routes_to_goal_distance_drops_by_one_each_step(layout):
    for goal in layout.buffer_cells:
        next_cell, distance = solve(layout, goal)
        for cell, step in next_cell.items():
            assert distance[cell] == distance[step] + 1


def test_routes_to_goal_routes_are_shortest(layout):
    # A distance is shortest iff it is 1 + the best distance among allowed next cells.
    graph = routes.build_graph(layout)
    for goal in layout.buffer_cells:
        _, distance = solve(layout, goal)
        for cell in layout.floor - {goal}:
            allowed = [v for v in graph[cell] if v == goal or v not in layout.buffer_cells]
            assert distance[cell] == 1 + min(distance[v] for v in allowed)


def test_routes_to_goal_routes_avoid_other_buffers(layout):
    for goal in layout.buffer_cells:
        next_cell, _ = solve(layout, goal)
        for start in layout.floor - {goal}:
            for cell in walk(next_cell, start, goal)[1:-1]:
                assert cell in layout.lanes, f"route {start} -> {goal} passes through {cell}"


def test_routes_to_goal_does_not_expand_other_buffers(mini):
    # (9, 9) can only be reached through K1, so it must not get a route to P2.
    reverse = dict(routes.reverse_graph(routes.build_graph(mini)))
    k1 = mini.buffers["K1"]
    reverse[k1] = (*reverse[k1], (9, 9))
    reverse[(9, 9)] = ()
    next_cell, distance = routes._routes_to_goal(mini, reverse, mini.buffers["P2"])
    assert k1 in distance
    assert (9, 9) not in distance
    assert (9, 9) not in next_cell


def test_routes_to_goal_does_not_change_the_reverse_graph(layout):
    # The same reverse graph is reused for every goal.
    reverse = routes.reverse_graph(routes.build_graph(layout))
    before = copy.deepcopy(reverse)
    for goal in layout.buffer_cells:
        routes._routes_to_goal(layout, reverse, goal)
    assert reverse == before


# Routes and build_routes


@pytest.fixture(scope="module")
def table(layout):
    return routes.build_routes(layout)


@pytest.fixture(scope="module")
def mini_table(mini):
    return routes.build_routes(mini)


@pytest.fixture(scope="module")
def main_table(main):
    return routes.build_routes(main)


def test_build_routes_stores_the_graph(layout, table):
    assert as_sets(table.graph) == as_sets(routes.build_graph(layout))


def test_build_routes_has_one_entry_per_buffer(layout, table):
    assert set(table.next_towards) == layout.buffer_cells
    assert set(table.distance) == layout.buffer_cells


def test_build_routes_matches_routes_to_goal(layout, table):
    for goal in layout.buffer_cells:
        next_cell, distance = solve(layout, goal)
        assert table.next_towards[goal] == next_cell
        assert table.distance[goal] == distance


def test_next_cell_mini(mini, mini_table):
    p1, p2 = mini.buffers["P1"], mini.buffers["P2"]
    assert mini_table.next_cell(mini.buffers["K2"], p2) == (1, 2)
    assert mini_table.next_cell((3, 2), p2) == p2
    # The intersection (2, 2) has two exits; the goal decides which one is used.
    assert mini_table.next_cell((2, 2), p2) == (2, 1)
    assert mini_table.next_cell((2, 2), p1) == (1, 2)


def test_route_mini_exact(mini, mini_table):
    route = mini_table.route(mini.buffers["K2"], mini.buffers["P2"])
    assert route == ((0, 2), (1, 2), (1, 1), (2, 1), (3, 1), (3, 2), (3, 3))


def test_route_is_a_tuple(mini, mini_table):
    assert isinstance(mini_table.route(mini.buffers["K2"], mini.buffers["P2"]), tuple)


def test_route_and_delta_from_goal_to_itself(layout, table):
    for goal in layout.buffer_cells:
        assert table.route(goal, goal) == (goal,)
        assert table.delta(goal, goal) == 0


def test_delta_mini(mini, mini_table):
    assert mini_table.delta(mini.buffers["K2"], mini.buffers["P2"]) == 6
    assert mini_table.delta(mini.buffers["P1"], mini.buffers["P2"]) == 6
    assert mini_table.delta((3, 2), mini.buffers["P2"]) == 1


@pytest.mark.parametrize(
    ("start", "goal", "moves"),
    [
        ("K1", "P3", 18),  # bottom lane, up the x = 10 connector, left to P3
        ("P1", "D1", 17),  # left and down around the loop, then the bottom lane
        ("P3", "D1", 10),  # through the x = 7 connector
        ("D3", "C1", 27),  # almost the whole outer loop
    ],
)
def test_delta_main_known_distances(main, main_table, start, goal, moves):
    assert main_table.delta(main.buffers[start], main.buffers[goal]) == moves


def test_delta_is_route_length_in_moves(layout, table):
    for goal in layout.buffer_cells:
        for start in layout.floor:
            assert table.delta(start, goal) == len(table.route(start, goal)) - 1


def test_route_starts_and_ends_in_the_right_cells(layout, table):
    for goal in layout.buffer_cells:
        for start in layout.floor:
            route = table.route(start, goal)
            assert route[0] == start
            assert route[-1] == goal


def test_route_follows_graph_edges(layout, table):
    for goal in layout.buffer_cells:
        for start in layout.floor:
            route = table.route(start, goal)
            for u, v in zip(route, route[1:], strict=False):
                assert v in table.graph[u], f"{u} -> {v} is not an edge"


def test_next_cell_is_second_cell_of_route(layout, table):
    for goal in layout.buffer_cells:
        for start in layout.floor - {goal}:
            assert table.next_cell(start, goal) == table.route(start, goal)[1]


def test_routes_are_suffix_consistent(layout, table):
    # After a move the rest of the route must not change (assumed by the proof of safe()).
    for goal in layout.buffer_cells:
        for start in layout.floor - {goal}:
            step = table.next_cell(start, goal)
            assert table.route(step, goal) == table.route(start, goal)[1:]


def test_every_buffer_reachable_from_every_buffer(layout, table):
    for start in layout.buffer_cells:
        for goal in layout.buffer_cells:
            assert table.route(start, goal)[-1] == goal


def test_routes_avoid_racks_and_other_buffers(layout, table):
    for goal in layout.buffer_cells:
        for start in layout.floor:
            route = table.route(start, goal)
            assert not set(route) & layout.racks
            assert set(route[1:-1]) <= layout.lanes, f"route {start} -> {goal}: {route}"
