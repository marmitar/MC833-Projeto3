"""
This type stub file was generated by pyright.
"""

from ._param_validation import Integral, Interval, validate_params

"""Graph utilities and algorithms."""
@validate_params({ "graph": ["array-like", "sparse matrix"],"source": [Interval(Integral, 0, None, closed="left")],"cutoff": [Interval(Integral, 0, None, closed="left"), None] }, prefer_skip_nested_validation=True)
def single_source_shortest_path_length(graph, source, *, cutoff=...): # -> dict[Any, Any]:
    """Return the length of the shortest path from source to all reachable nodes.

    Parameters
    ----------
    graph : {array-like, sparse matrix} of shape (n_nodes, n_nodes)
        Adjacency matrix of the graph. Sparse matrix of format LIL is
        preferred.

    source : int
       Start node for path.

    cutoff : int, default=None
        Depth to stop the search - only paths of length <= cutoff are returned.

    Returns
    -------
    paths : dict
        Reachable end nodes mapped to length of path from source,
        i.e. `{end: path_length}`.

    Examples
    --------
    >>> from sklearn.utils.graph import single_source_shortest_path_length
    >>> import numpy as np
    >>> graph = np.array([[ 0, 1, 0, 0],
    ...                   [ 1, 0, 1, 0],
    ...                   [ 0, 1, 0, 0],
    ...                   [ 0, 0, 0, 0]])
    >>> single_source_shortest_path_length(graph, 0)
    {0: 0, 1: 1, 2: 2}
    >>> graph = np.ones((6, 6))
    >>> sorted(single_source_shortest_path_length(graph, 2).items())
    [(0, 1), (1, 1), (2, 0), (3, 1), (4, 1), (5, 1)]
    """
    ...
