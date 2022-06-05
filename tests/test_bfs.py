import pytest
import pygraphblas as gb

from project.bfs import bfs, multi_bfs


@pytest.fixture
def sample_graph():
    return gb.matrix.Matrix.from_lists(
        [0, 0, 1, 1, 3, 4, 4], [1, 4, 2, 4, 4, 1, 3], [1, 1, 1, 1, 1, 1, 1]
    )


@pytest.fixture
def noedges_graph():
    return gb.matrix.Matrix.dense(gb.INT32, 5, 5)


@pytest.fixture
def pseudo_graph():
    return gb.matrix.Matrix.from_lists([0, 0, 1], [0, 1, 1], [1, 1, 1])


def test_noedges_bfs(noedges_graph):
    expected_distances = [-1 for _ in range(noedges_graph.nrows)]
    expected_distances[0] = 0
    assert bfs(noedges_graph, 0) == expected_distances


def test_pseudo_graph_bfs(pseudo_graph):
    assert bfs(pseudo_graph, 0) == [0, 1]


@pytest.mark.parametrize(
    "start_node, expected_distances",
    [(1, [-1, 0, 1, 2, 1]), (2, [-1, -1, 0, -1, -1]), (0, [0, 1, 2, 2, 1])],
)
def test_single_bfs(sample_graph, start_node, expected_distances):
    assert bfs(sample_graph, start_node) == expected_distances


@pytest.mark.parametrize(
    "start_nodes, expected_distances",
    [
        (
            [1, 2, 0],
            [(1, [-1, 0, 1, 2, 1]), (2, [-1, -1, 0, -1, -1]), (0, [0, 1, 2, 2, 1])],
        )
    ],
)
def test_multi_bfs(sample_graph, start_nodes, expected_distances):
    assert multi_bfs(sample_graph, start_nodes) == expected_distances


def test_big_start(sample_graph):
    with pytest.raises(ValueError):
        bfs(sample_graph, 100)


def test_negative_start(sample_graph):
    with pytest.raises(ValueError):
        bfs(sample_graph, -6)


def test_multi_wrong_start(sample_graph):
    with pytest.raises(ValueError):
        multi_bfs(sample_graph, [1, 2, 3, 40])
