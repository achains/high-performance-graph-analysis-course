import pygraphblas as gb

__all__ = ["triangle_count"]


def triangle_count(matrix: gb.Matrix):
    """
    For each node get number of triangles in which it participates.

    Parameters
    ----------
    matrix: gb.Matrix
        Graph adjacency matrix

    Returns
    -------
    triangle_number: list
        triangle_number[i] = <number of triangles in which node[i] participates>
    """
    if not matrix.square:
        raise ValueError("Expected square adjacency matrix")

    if matrix.nrows == 0:
        return []

    matrix = matrix + matrix.transpose()
    matrix.assign_scalar(False, mask=matrix.diag())

    three_size_path = matrix.mxm(
        matrix, cast=gb.INT64, accum=gb.INT64.PLUS, mask=matrix
    )
    three_size_path = three_size_path.reduce_vector() / 2

    result = [0 for _ in range(matrix.nrows)]
    nodes, tri_numbers = three_size_path.to_lists()

    for i, node in enumerate(nodes):
        result[node] = tri_numbers[i]

    return result
