import pygraphblas as gb

__all__ = ["bfs", "multi_bfs"]


def _bfs(matrix: gb.Matrix, start_node_list):
    if not matrix.square:
        raise ValueError("Expected square adjacency matrix")
    if any(start >= matrix.nrows or start < 0 for start in start_node_list):
        raise ValueError(
            f"Start node should be a non-negative value less than {matrix.nrows}"
        )
    front = gb.Matrix.sparse(gb.BOOL, nrows=len(start_node_list), ncols=matrix.ncols)
    visited = gb.Matrix.sparse(gb.BOOL, nrows=len(start_node_list), ncols=matrix.ncols)
    distances = gb.Matrix.dense(
        gb.INT64, nrows=len(start_node_list), ncols=matrix.ncols, fill=-1
    )

    for i, j in enumerate(start_node_list):
        front.assign_scalar(True, i, j)
        visited.assign_scalar(True, i, j)
        distances.assign_scalar(0, i, j)

    level = 1
    is_changed = True
    while level <= matrix.nrows and is_changed:
        prev_nnz = visited.nvals
        front.mxm(matrix, mask=visited, out=front, desc=gb.descriptor.RC)
        visited.eadd(front, front.type.lxor_monoid, out=visited, desc=gb.descriptor.R)
        distances.assign_scalar(level, mask=front)
        level += 1
        if visited.nvals == prev_nnz:
            is_changed = False

    return [(node, list(distances[i].vals)) for i, node in enumerate(start_node_list)]


def bfs(matrix: gb.Matrix, start: int) -> list:
    """Breadth First Search.
    Find distances from start node to each graph node

    Parameters
    ----------
    matrix: gb.Matrix
        Graph adjacency matrix
    start: int
        Search start node

    Returns
    -------
    D: list
        Distances list D,
        D[i] == -1 if i-th node is unreachable from start
    """
    return _bfs(matrix, [start])[0][1]


def multi_bfs(matrix: gb.Matrix, start_node_list: list):
    """Breadth First Search.
    Find distances from start node list to each graph node

    Parameters
    ----------
    matrix: gb.Matrix
        Graph adjacency matrix
    start_node_list: list
        Search start node list

    Returns
    -------
    D: list[list]
        Distances matrix D,
        -1 if i-th node is unreachable from corresponding start node
    """
    return _bfs(matrix, start_node_list)
