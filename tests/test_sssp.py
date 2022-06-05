import pytest
import pygraphblas as gb

from project.sssp import sssp, mssp


@pytest.fixture
def sample_graph():
    return gb.Matrix.from_lists(
        [0, 0, 1, 3, 3, 4, 1, 5],
        [1, 3, 2, 4, 5, 2, 5, 4],
        [9, 3, 8, 6, 1, 4, 7, 2],
    )


@pytest.fixture
def pseudo_graph():
    return gb.Matrix.from_lists(
        [0, 0, 0, 1, 2, 3, 3], [0, 0, 1, 0, 3, 2, 3], [1, 1, 2, 3, 4, 5, 6]
    )


@pytest.fixture()
def single_node_graph():
    return gb.Matrix.dense(gb.INT32, 1, 1)


def test_pseudo_sssp(pseudo_graph):
    assert sssp(pseudo_graph, 0) == [0, 2, -1, -1]


def test_single_node_sssp(single_node_graph):
    assert sssp(single_node_graph, 0) == [0]


@pytest.mark.parametrize(
    "I, J, V, size, start_vertex, expected_ans",
    [
        (
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [0, 1, 2, 0, 1, 2, 0, 1, 2],
            [0.0, 3.0, 1.0, 5000.0, 5000.0, 5000.0, 5000.0, 1.0, 5000.0],
            3,
            0,
            [0.0, 2.0, 1.0],
        ),
        (
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [0, 1, 2, 0, 1, 2, 0, 1, 2],
            [0, 3, 1, 5000, 5000, 5000, 5000, 1, 5000],
            4,
            0,
            [0.0, 2.0, 1.0, -1],
        ),
        ([0], [1], [5], 5, 0, [0, 5, -1, -1, -1]),
    ],
)
def test_sssp(I, J, V, size, start_vertex, expected_ans):
    adj_matrix = gb.Matrix.from_lists(I, J, V, nrows=size, ncols=size)
    assert sssp(adj_matrix, start_vertex) == expected_ans


@pytest.mark.parametrize(
    "start_nodes, expected_distances",
    [
        (
            [1, 2, 0],
            [
                (1, [-1, 0, 8, -1, 9, 7]),
                (2, [-1, -1, 0, -1, -1, -1]),
                (0, [0, 9, 10, 3, 6, 4]),
            ],
        )
    ],
)
def test_mssp(sample_graph, start_nodes, expected_distances):
    assert mssp(sample_graph, start_nodes) == expected_distances


@pytest.mark.parametrize(
    "I, J, V, size, start_vertices, expected_ans",
    [
        (
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [0, 1, 2, 0, 1, 2, 0, 1, 2],
            [0.0, 3.0, 1.0, 5000.0, 5000.0, 5000.0, 5000.0, 1.0, 5000.0],
            3,
            [0, 1, 2],
            [(0, [0.0, 2.0, 1.0]), (1, [5000.0, 0.0, 5000.0]), (2, [5000.0, 1.0, 0.0])],
        ),
        (
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [0, 1, 2, 0, 1, 2, 0, 1, 2],
            [0, 3, 1, 5000, 5000, 5000, 5000, 1, 5000],
            4,
            [0, 1, 2],
            [
                (0, [0.0, 2.0, 1.0, -1]),
                (1, [5000.0, 0.0, 5000.0, -1]),
                (2, [5000.0, 1.0, 0.0, -1]),
            ],
        ),
    ],
)
def test_mssp_float_w(I, J, V, size, start_vertices, expected_ans):
    adj_matrix = gb.Matrix.from_lists(I, J, V, nrows=size, ncols=size)
    assert mssp(adj_matrix, start_vertices) == expected_ans


def test_big_start(sample_graph):
    with pytest.raises(ValueError):
        sssp(sample_graph, 100)


def test_negative_start(sample_graph):
    with pytest.raises(ValueError):
        sssp(sample_graph, -6)


def test_multi_wrong_start(sample_graph):
    with pytest.raises(ValueError):
        mssp(sample_graph, [1, 2, 3, 40])
