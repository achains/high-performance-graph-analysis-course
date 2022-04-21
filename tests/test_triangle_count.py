import pytest
import pygraphblas as gb

from project.triangle_count import triangle_count


@pytest.fixture
def sample_graph():
    return gb.Matrix.from_lists(
        [0, 1, 1, 2, 0, 2, 1, 3, 2, 3, 4, 3, 2, 4, 5, 4],
        [1, 0, 2, 1, 2, 0, 3, 1, 3, 2, 3, 4, 4, 2, 4, 5],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    )


@pytest.fixture
def notri_graph():
    return gb.Matrix.from_lists(
        [0, 1, 1, 2],
        [1, 0, 2, 1],
        [1, 1, 1, 1]
    )


@pytest.fixture
def pseudo_graph():
    return gb.Matrix.from_lists(
        [0, 0, 1, 1, 2, 0, 2, 1, 3, 2, 3, 4, 3, 2, 4, 5, 4],
        [0, 1, 0, 2, 1, 2, 0, 3, 1, 3, 2, 3, 4, 4, 2, 4, 5],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    )


def test_triangles(sample_graph):
    expected = [1, 2, 3, 2, 1, 0]
    assert triangle_count(sample_graph) == expected


def test_no_triangles(notri_graph):
    expected = [0, 0, 0]
    assert triangle_count(notri_graph) == expected


def test_pseudo(pseudo_graph):
    expected = [1, 2, 3, 2, 1, 0]
    assert triangle_count(pseudo_graph) == expected
