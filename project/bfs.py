import pygraphblas as gb


def bfs(matrix: gb.Matrix, start: int):
    """Breadth First Search.
    Find distances to graph each node

    Parameters
    ----------
    matrix: gb.Matrix
        Graph adjacency matrix
    start: int
        Search start nodes

    Returns
    -------
    distances: gb.Vector
        ddd
    """
    distances = gb.Vector.sparse(gb.UINT8, matrix.nrows)
    visited = gb.Vector.sparse(gb.BOOL, matrix.nrows)

    visited[start] = True
    level = 1

    while visited.reduce_bool() and level <= matrix.nrows:
        distances.assign_scalar(level, mask=visited)
        distances.vxm(matrix, mask=distances, out=visited, desc=gb.descriptor.RC)
        level += 1

    return distances
