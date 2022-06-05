import pygraphblas as gb
import numpy as np

__all__ = ["sssp", "mssp"]


def _mssp(matrix: gb.Matrix, start_node_list: list):
    if not matrix.square:
        raise ValueError("Expected square adjacency matrix")
    if any(start >= matrix.nrows or start < 0 for start in start_node_list):
        raise ValueError(
            f"Start node should be a non-negative value less than {matrix.nrows}"
        )

    int_max = np.iinfo(np.int32).max

    v = gb.Matrix.dense(
        matrix.type, nrows=matrix.nrows, ncols=matrix.ncols, fill=int_max
    )
    for i, j in enumerate(start_node_list):
        v.assign_scalar(0, i, j)

    is_changed = True
    i = 0
    while i < matrix.nrows and is_changed:
        prev_nnz = v.nvals
        v.min_plus(matrix, out=v, accum=gb.INT64.min)
        is_changed = prev_nnz == v.nvals
        i += 1

    result = []
    for i, node in enumerate(start_node_list):
        line = v[i]
        line.assign_scalar(-1, mask=(v[i] == int_max))
        result.append((node, list(line.vals)))

    return result


def sssp(matrix: gb.Matrix, start_node: int):
    """
    Single Source Shortest Path (SSSP)

    Parameters
    ----------
    matrix: gb.Matrix
        Graph adjacency matrix
    start_node: int
        Search start node

    Returns
    -------
    D: list
        Distances list D,
        D[i] == -1 if i-th node is unreachable from start
    """
    return _mssp(matrix, [start_node])[0][1]


def mssp(matrix: gb.Matrix, start_node_list: list):
    """
    Multi Source Shortest Path (SSSP)

    Parameters
    ----------
    matrix: gb.Matrix
        Graph adjacency matrix
    start_node_list: list
        Search start nodes

    Returns
    -------
    D: list
        Distances list D,
        D[i] == -1 if i-th node is unreachable from start
    """
    return _mssp(matrix, start_node_list)
