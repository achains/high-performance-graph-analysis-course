import pytest
import pygraphblas as gb

from project.triangle_count import triangle_count


@pytest.fixture
def sample_graph():
    return gb.Matrix.from_lists(
        [0, 1, 1, 2, 0, 2, 1, 3, 2, 3, 4, 3, 2, 4, 5, 4],
        [1, 0, 2, 1, 2, 0, 3, 1, 3, 2, 3, 4, 4, 2, 4, 5],
        [True for _ in range(16)],
    )


@pytest.fixture
def notri_graph():
    return gb.Matrix.from_lists([0, 1, 1, 2], [1, 0, 2, 1], [True, True, True, True])


@pytest.fixture
def single_node_graph():
    return gb.Matrix.dense(gb.INT32, 1, 1)


@pytest.fixture
def empty_graph():
    return gb.Matrix.identity(gb.INT32, 0, 0)


@pytest.fixture
def pseudo_graph():
    return gb.Matrix.from_lists(
        [0, 0, 1, 1, 2, 0, 2, 1, 3, 2, 3, 4, 3, 2, 4, 5, 4],
        [0, 1, 0, 2, 1, 2, 0, 3, 1, 3, 2, 3, 4, 4, 2, 4, 5],
        [True for _ in range(17)],
    )


def test_single_node(single_node_graph):
    assert triangle_count(single_node_graph) == [0]


def test_empty_graph(empty_graph):
    assert triangle_count(empty_graph) == []


def test_triangles(sample_graph):
    expected = [1, 2, 3, 2, 1, 0]
    assert triangle_count(sample_graph) == expected


def test_no_triangles(notri_graph):
    expected = [0, 0, 0]
    assert triangle_count(notri_graph) == expected


def test_pseudo(pseudo_graph):
    expected = [1, 2, 3, 2, 1, 0]
    assert triangle_count(pseudo_graph) == expected
