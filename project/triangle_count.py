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

    three_size_path = matrix
    # Remove loops
    three_size_path.assign_scalar(0, mask=three_size_path.diag())

    for _ in range(2):
        three_size_path = matrix.mxm(
            three_size_path, cast=gb.types.INT64, accum=gb.types.INT64.PLUS
        )

    three_size_path = three_size_path.diag().reduce_vector()
    three_size_path /= 2

    result = [0 for _ in range(matrix.nrows)]
    nodes, tri_numbers = three_size_path.to_lists()

    for i, node in enumerate(nodes):
        result[node] = tri_numbers[i]

    return result
