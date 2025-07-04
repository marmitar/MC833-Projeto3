"""
This type stub file was generated by pyright.
"""

import numpy as np
from numbers import Integral, Real
from ..base import BaseEstimator, ClusterMixin, _fit_context
from ..metrics.pairwise import _VALID_METRICS
from ..utils._param_validation import Interval, RealNotInt, StrOptions, validate_params

"""Ordering Points To Identify the Clustering Structure (OPTICS)

These routines execute the OPTICS algorithm, and implement various
cluster extraction methods of the ordered list.
"""
class OPTICS(ClusterMixin, BaseEstimator):
    """Estimate clustering structure from vector array.

    OPTICS (Ordering Points To Identify the Clustering Structure), closely
    related to DBSCAN, finds core samples of high density and expands clusters
    from them [1]_. Unlike DBSCAN, it keeps cluster hierarchy for a variable
    neighborhood radius. Better suited for usage on large datasets than the
    current scikit-learn implementation of DBSCAN.

    Clusters are then extracted from the cluster-order using a
    DBSCAN-like method (cluster_method = 'dbscan') or an automatic
    technique proposed in [1]_ (cluster_method = 'xi').

    This implementation deviates from the original OPTICS by first performing
    k-nearest-neighborhood searches on all points to identify core sizes of
    all points (instead of computing neighbors while looping through points).
    Reachability distances to only unprocessed points are then computed, to
    construct the cluster order, similar to the original OPTICS.
    Note that we do not employ a heap to manage the expansion
    candidates, so the time complexity will be O(n^2).

    Read more in the :ref:`User Guide <optics>`.

    Parameters
    ----------
    min_samples : int > 1 or float between 0 and 1, default=5
        The number of samples in a neighborhood for a point to be considered as
        a core point. Also, up and down steep regions can't have more than
        ``min_samples`` consecutive non-steep points. Expressed as an absolute
        number or a fraction of the number of samples (rounded to be at least
        2).

    max_eps : float, default=np.inf
        The maximum distance between two samples for one to be considered as
        in the neighborhood of the other. Default value of ``np.inf`` will
        identify clusters across all scales; reducing ``max_eps`` will result
        in shorter run times.

    metric : str or callable, default='minkowski'
        Metric to use for distance computation. Any metric from scikit-learn
        or :mod:`scipy.spatial.distance` can be used.

        If `metric` is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays as input and return one value indicating the
        distance between them. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string. If metric is
        "precomputed", `X` is assumed to be a distance matrix and must be
        square.

        Valid values for metric are:

        - from scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
          'manhattan']

        - from scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
          'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski',
          'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao',
          'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
          'yule']

        Sparse matrices are only supported by scikit-learn metrics.
        See :mod:`scipy.spatial.distance` for details on these metrics.

        .. note::
           `'kulsinski'` is deprecated from SciPy 1.9 and will be removed in SciPy 1.11.

    p : float, default=2
        Parameter for the Minkowski metric from
        :class:`~sklearn.metrics.pairwise_distances`. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.

    cluster_method : {'xi', 'dbscan'}, default='xi'
        The extraction method used to extract clusters using the calculated
        reachability and ordering.

    eps : float, default=None
        The maximum distance between two samples for one to be considered as
        in the neighborhood of the other. By default it assumes the same value
        as ``max_eps``.
        Used only when ``cluster_method='dbscan'``.

    xi : float between 0 and 1, default=0.05
        Determines the minimum steepness on the reachability plot that
        constitutes a cluster boundary. For example, an upwards point in the
        reachability plot is defined by the ratio from one point to its
        successor being at most 1-xi.
        Used only when ``cluster_method='xi'``.

    predecessor_correction : bool, default=True
        Correct clusters according to the predecessors calculated by OPTICS
        [2]_. This parameter has minimal effect on most datasets.
        Used only when ``cluster_method='xi'``.

    min_cluster_size : int > 1 or float between 0 and 1, default=None
        Minimum number of samples in an OPTICS cluster, expressed as an
        absolute number or a fraction of the number of samples (rounded to be
        at least 2). If ``None``, the value of ``min_samples`` is used instead.
        Used only when ``cluster_method='xi'``.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        Algorithm used to compute the nearest neighbors:

        - 'ball_tree' will use :class:`~sklearn.neighbors.BallTree`.
        - 'kd_tree' will use :class:`~sklearn.neighbors.KDTree`.
        - 'brute' will use a brute-force search.
        - 'auto' (default) will attempt to decide the most appropriate
          algorithm based on the values passed to :meth:`fit` method.

        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.

    leaf_size : int, default=30
        Leaf size passed to :class:`~sklearn.neighbors.BallTree` or
        :class:`~sklearn.neighbors.KDTree`. This can affect the speed of the
        construction and query, as well as the memory required to store the
        tree. The optimal value depends on the nature of the problem.

    memory : str or object with the joblib.Memory interface, default=None
        Used to cache the output of the computation of the tree.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels for each point in the dataset given to fit().
        Noisy samples and points which are not included in a leaf cluster
        of ``cluster_hierarchy_`` are labeled as -1.

    reachability_ : ndarray of shape (n_samples,)
        Reachability distances per sample, indexed by object order. Use
        ``clust.reachability_[clust.ordering_]`` to access in cluster order.

    ordering_ : ndarray of shape (n_samples,)
        The cluster ordered list of sample indices.

    core_distances_ : ndarray of shape (n_samples,)
        Distance at which each sample becomes a core point, indexed by object
        order. Points which will never be core have a distance of inf. Use
        ``clust.core_distances_[clust.ordering_]`` to access in cluster order.

    predecessor_ : ndarray of shape (n_samples,)
        Point that a sample was reached from, indexed by object order.
        Seed points have a predecessor of -1.

    cluster_hierarchy_ : ndarray of shape (n_clusters, 2)
        The list of clusters in the form of ``[start, end]`` in each row, with
        all indices inclusive. The clusters are ordered according to
        ``(end, -start)`` (ascending) so that larger clusters encompassing
        smaller clusters come after those smaller ones. Since ``labels_`` does
        not reflect the hierarchy, usually
        ``len(cluster_hierarchy_) > np.unique(optics.labels_)``. Please also
        note that these indices are of the ``ordering_``, i.e.
        ``X[ordering_][start:end + 1]`` form a cluster.
        Only available when ``cluster_method='xi'``.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    DBSCAN : A similar clustering for a specified neighborhood radius (eps).
        Our implementation is optimized for runtime.

    References
    ----------
    .. [1] Ankerst, Mihael, Markus M. Breunig, Hans-Peter Kriegel,
       and Jörg Sander. "OPTICS: ordering points to identify the clustering
       structure." ACM SIGMOD Record 28, no. 2 (1999): 49-60.

    .. [2] Schubert, Erich, Michael Gertz.
       "Improving the Cluster Structure Extracted from OPTICS Plots." Proc. of
       the Conference "Lernen, Wissen, Daten, Analysen" (LWDA) (2018): 318-329.

    Examples
    --------
    >>> from sklearn.cluster import OPTICS
    >>> import numpy as np
    >>> X = np.array([[1, 2], [2, 5], [3, 6],
    ...               [8, 7], [8, 8], [7, 3]])
    >>> clustering = OPTICS(min_samples=2).fit(X)
    >>> clustering.labels_
    array([0, 0, 0, 1, 1, 1])

    For a more detailed example see
    :ref:`sphx_glr_auto_examples_cluster_plot_optics.py`.

    For a comparison of OPTICS with other clustering algorithms, see
    :ref:`sphx_glr_auto_examples_cluster_plot_cluster_comparison.py`
    """
    _parameter_constraints: dict = ...
    def __init__(self, *, min_samples=..., max_eps=..., metric=..., p=..., metric_params=..., cluster_method=..., eps=..., xi=..., predecessor_correction=..., min_cluster_size=..., algorithm=..., leaf_size=..., memory=..., n_jobs=...) -> None:
        ...

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, y=...): # -> Self:
        """Perform OPTICS clustering.

        Extracts an ordered list of points and reachability distances, and
        performs initial clustering using ``max_eps`` distance specified at
        OPTICS object instantiation.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features), or \
                (n_samples, n_samples) if metric='precomputed'
            A feature array, or array of distances between samples if
            metric='precomputed'. If a sparse matrix is provided, it will be
            converted into CSR format.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns a fitted instance of self.
        """
        ...



@validate_params({ "X": [np.ndarray, "sparse matrix"],"min_samples": [Interval(Integral, 2, None, closed="left"), Interval(RealNotInt, 0, 1, closed="both")],"max_eps": [Interval(Real, 0, None, closed="both")],"metric": [StrOptions(set(_VALID_METRICS) | "precomputed"), callable],"p": [Interval(Real, 0, None, closed="right"), None],"metric_params": [dict, None],"algorithm": [StrOptions("auto", "brute", "ball_tree", "kd_tree")],"leaf_size": [Interval(Integral, 1, None, closed="left")],"n_jobs": [Integral, None] }, prefer_skip_nested_validation=False)
def compute_optics_graph(X, *, min_samples, max_eps, metric, p, metric_params, algorithm, leaf_size, n_jobs): # -> tuple[_Array1D[Any], _Array1D[float64], _Array1D[float64], _Array1D[Any]]:
    """Compute the OPTICS reachability graph.

    Read more in the :ref:`User Guide <optics>`.

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features), or \
            (n_samples, n_samples) if metric='precomputed'
        A feature array, or array of distances between samples if
        metric='precomputed'.

    min_samples : int > 1 or float between 0 and 1
        The number of samples in a neighborhood for a point to be considered
        as a core point. Expressed as an absolute number or a fraction of the
        number of samples (rounded to be at least 2).

    max_eps : float, default=np.inf
        The maximum distance between two samples for one to be considered as
        in the neighborhood of the other. Default value of ``np.inf`` will
        identify clusters across all scales; reducing ``max_eps`` will result
        in shorter run times.

    metric : str or callable, default='minkowski'
        Metric to use for distance computation. Any metric from scikit-learn
        or scipy.spatial.distance can be used.

        If metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays as input and return one value indicating the
        distance between them. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string. If metric is
        "precomputed", X is assumed to be a distance matrix and must be square.

        Valid values for metric are:

        - from scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
          'manhattan']

        - from scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
          'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski',
          'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao',
          'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
          'yule']

        See the documentation for scipy.spatial.distance for details on these
        metrics.

        .. note::
           `'kulsinski'` is deprecated from SciPy 1.9 and will be removed in SciPy 1.11.

    p : float, default=2
        Parameter for the Minkowski metric from
        :class:`~sklearn.metrics.pairwise_distances`. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        Algorithm used to compute the nearest neighbors:

        - 'ball_tree' will use :class:`~sklearn.neighbors.BallTree`.
        - 'kd_tree' will use :class:`~sklearn.neighbors.KDTree`.
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to `fit` method. (default)

        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.

    leaf_size : int, default=30
        Leaf size passed to :class:`~sklearn.neighbors.BallTree` or
        :class:`~sklearn.neighbors.KDTree`. This can affect the speed of the
        construction and query, as well as the memory required to store the
        tree. The optimal value depends on the nature of the problem.

    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Returns
    -------
    ordering_ : array of shape (n_samples,)
        The cluster ordered list of sample indices.

    core_distances_ : array of shape (n_samples,)
        Distance at which each sample becomes a core point, indexed by object
        order. Points which will never be core have a distance of inf. Use
        ``clust.core_distances_[clust.ordering_]`` to access in cluster order.

    reachability_ : array of shape (n_samples,)
        Reachability distances per sample, indexed by object order. Use
        ``clust.reachability_[clust.ordering_]`` to access in cluster order.

    predecessor_ : array of shape (n_samples,)
        Point that a sample was reached from, indexed by object order.
        Seed points have a predecessor of -1.

    References
    ----------
    .. [1] Ankerst, Mihael, Markus M. Breunig, Hans-Peter Kriegel,
       and Jörg Sander. "OPTICS: ordering points to identify the clustering
       structure." ACM SIGMOD Record 28, no. 2 (1999): 49-60.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.cluster import compute_optics_graph
    >>> X = np.array([[1, 2], [2, 5], [3, 6],
    ...               [8, 7], [8, 8], [7, 3]])
    >>> ordering, core_distances, reachability, predecessor = compute_optics_graph(
    ...     X,
    ...     min_samples=2,
    ...     max_eps=np.inf,
    ...     metric="minkowski",
    ...     p=2,
    ...     metric_params=None,
    ...     algorithm="auto",
    ...     leaf_size=30,
    ...     n_jobs=None,
    ... )
    >>> ordering
    array([0, 1, 2, 5, 3, 4])
    >>> core_distances
    array([3.16, 1.41, 1.41, 1.        , 1.        ,
           4.12])
    >>> reachability
    array([       inf, 3.16, 1.41, 4.12, 1.        ,
           5.        ])
    >>> predecessor
    array([-1,  0,  1,  5,  3,  2])
    """
    ...

@validate_params({ "reachability": [np.ndarray],"core_distances": [np.ndarray],"ordering": [np.ndarray],"eps": [Interval(Real, 0, None, closed="both")] }, prefer_skip_nested_validation=True)
def cluster_optics_dbscan(*, reachability, core_distances, ordering, eps): # -> _Array1D[Any]:
    """Perform DBSCAN extraction for an arbitrary epsilon.

    Extracting the clusters runs in linear time. Note that this results in
    ``labels_`` which are close to a :class:`~sklearn.cluster.DBSCAN` with
    similar settings and ``eps``, only if ``eps`` is close to ``max_eps``.

    Parameters
    ----------
    reachability : ndarray of shape (n_samples,)
        Reachability distances calculated by OPTICS (``reachability_``).

    core_distances : ndarray of shape (n_samples,)
        Distances at which points become core (``core_distances_``).

    ordering : ndarray of shape (n_samples,)
        OPTICS ordered point indices (``ordering_``).

    eps : float
        DBSCAN ``eps`` parameter. Must be set to < ``max_eps``. Results
        will be close to DBSCAN algorithm if ``eps`` and ``max_eps`` are close
        to one another.

    Returns
    -------
    labels_ : array of shape (n_samples,)
        The estimated labels.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.cluster import cluster_optics_dbscan, compute_optics_graph
    >>> X = np.array([[1, 2], [2, 5], [3, 6],
    ...               [8, 7], [8, 8], [7, 3]])
    >>> ordering, core_distances, reachability, predecessor = compute_optics_graph(
    ...     X,
    ...     min_samples=2,
    ...     max_eps=np.inf,
    ...     metric="minkowski",
    ...     p=2,
    ...     metric_params=None,
    ...     algorithm="auto",
    ...     leaf_size=30,
    ...     n_jobs=None,
    ... )
    >>> eps = 4.5
    >>> labels = cluster_optics_dbscan(
    ...     reachability=reachability,
    ...     core_distances=core_distances,
    ...     ordering=ordering,
    ...     eps=eps,
    ... )
    >>> labels
    array([0, 0, 0, 1, 1, 1])
    """
    ...

@validate_params({ "reachability": [np.ndarray],"predecessor": [np.ndarray],"ordering": [np.ndarray],"min_samples": [Interval(Integral, 2, None, closed="left"), Interval(RealNotInt, 0, 1, closed="both")],"min_cluster_size": [Interval(Integral, 2, None, closed="left"), Interval(RealNotInt, 0, 1, closed="both"), None],"xi": [Interval(Real, 0, 1, closed="both")],"predecessor_correction": ["boolean"] }, prefer_skip_nested_validation=True)
def cluster_optics_xi(*, reachability, predecessor, ordering, min_samples, min_cluster_size=..., xi=..., predecessor_correction=...): # -> tuple[_Array[tuple[int], Any], NDArray[Any]]:
    """Automatically extract clusters according to the Xi-steep method.

    Parameters
    ----------
    reachability : ndarray of shape (n_samples,)
        Reachability distances calculated by OPTICS (`reachability_`).

    predecessor : ndarray of shape (n_samples,)
        Predecessors calculated by OPTICS.

    ordering : ndarray of shape (n_samples,)
        OPTICS ordered point indices (`ordering_`).

    min_samples : int > 1 or float between 0 and 1
        The same as the min_samples given to OPTICS. Up and down steep regions
        can't have more then ``min_samples`` consecutive non-steep points.
        Expressed as an absolute number or a fraction of the number of samples
        (rounded to be at least 2).

    min_cluster_size : int > 1 or float between 0 and 1, default=None
        Minimum number of samples in an OPTICS cluster, expressed as an
        absolute number or a fraction of the number of samples (rounded to be
        at least 2). If ``None``, the value of ``min_samples`` is used instead.

    xi : float between 0 and 1, default=0.05
        Determines the minimum steepness on the reachability plot that
        constitutes a cluster boundary. For example, an upwards point in the
        reachability plot is defined by the ratio from one point to its
        successor being at most 1-xi.

    predecessor_correction : bool, default=True
        Correct clusters based on the calculated predecessors.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        The labels assigned to samples. Points which are not included
        in any cluster are labeled as -1.

    clusters : ndarray of shape (n_clusters, 2)
        The list of clusters in the form of ``[start, end]`` in each row, with
        all indices inclusive. The clusters are ordered according to ``(end,
        -start)`` (ascending) so that larger clusters encompassing smaller
        clusters come after such nested smaller clusters. Since ``labels`` does
        not reflect the hierarchy, usually ``len(clusters) >
        np.unique(labels)``.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.cluster import cluster_optics_xi, compute_optics_graph
    >>> X = np.array([[1, 2], [2, 5], [3, 6],
    ...               [8, 7], [8, 8], [7, 3]])
    >>> ordering, core_distances, reachability, predecessor = compute_optics_graph(
    ...     X,
    ...     min_samples=2,
    ...     max_eps=np.inf,
    ...     metric="minkowski",
    ...     p=2,
    ...     metric_params=None,
    ...     algorithm="auto",
    ...     leaf_size=30,
    ...     n_jobs=None
    ... )
    >>> min_samples = 2
    >>> labels, clusters = cluster_optics_xi(
    ...     reachability=reachability,
    ...     predecessor=predecessor,
    ...     ordering=ordering,
    ...     min_samples=min_samples,
    ... )
    >>> labels
    array([0, 0, 0, 1, 1, 1])
    >>> clusters
    array([[0, 2],
           [3, 5],
           [0, 5]])
    """
    ...
