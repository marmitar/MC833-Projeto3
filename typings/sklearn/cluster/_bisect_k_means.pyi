"""
This type stub file was generated by pyright.
"""

from ..base import _fit_context
from ._kmeans import _BaseKMeans

"""Bisecting K-means clustering."""
class _BisectingTree:
    """Tree structure representing the hierarchical clusters of BisectingKMeans."""
    def __init__(self, center, indices, score) -> None:
        """Create a new cluster node in the tree.

        The node holds the center of this cluster and the indices of the data points
        that belong to it.
        """
        ...

    def split(self, labels, centers, scores): # -> None:
        """Split the cluster node into two subclusters."""
        ...

    def get_cluster_to_bisect(self): # -> Self:
        """Return the cluster node to bisect next.

        It's based on the score of the cluster, which can be either the number of
        data points assigned to that cluster or the inertia of that cluster
        (see `bisecting_strategy` for details).
        """
        ...

    def iter_leaves(self): # -> Generator[Self | Any, Any, None]:
        """Iterate over all the cluster leaves in the tree."""
        ...



class BisectingKMeans(_BaseKMeans):
    """Bisecting K-Means clustering.

    Read more in the :ref:`User Guide <bisect_k_means>`.

    .. versionadded:: 1.1

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.

    init : {'k-means++', 'random'} or callable, default='random'
        Method for initialization:

        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.

        'random': choose `n_clusters` observations (rows) at random from data
        for the initial centroids.

        If a callable is passed, it should take arguments X, n_clusters and a
        random state and return an initialization.

    n_init : int, default=1
        Number of time the inner k-means algorithm will be run with different
        centroid seeds in each bisection.
        That will result producing for each bisection best output of n_init
        consecutive runs in terms of inertia.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization
        in inner K-Means. Use an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    max_iter : int, default=300
        Maximum number of iterations of the inner k-means algorithm at each
        bisection.

    verbose : int, default=0
        Verbosity mode.

    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations  to declare
        convergence. Used in inner k-means algorithm at each bisection to pick
        best possible clusters.

    copy_x : bool, default=True
        When pre-computing distances it is more numerically accurate to center
        the data first. If copy_x is True (default), then the original data is
        not modified. If False, the original data is modified, and put back
        before the function returns, but small numerical differences may be
        introduced by subtracting and then adding the data mean. Note that if
        the original data is not C-contiguous, a copy will be made even if
        copy_x is False. If the original data is sparse, but not in CSR format,
        a copy will be made even if copy_x is False.

    algorithm : {"lloyd", "elkan"}, default="lloyd"
        Inner K-means algorithm used in bisection.
        The classical EM-style algorithm is `"lloyd"`.
        The `"elkan"` variation can be more efficient on some datasets with
        well-defined clusters, by using the triangle inequality. However it's
        more memory intensive due to the allocation of an extra array of shape
        `(n_samples, n_clusters)`.

    bisecting_strategy : {"biggest_inertia", "largest_cluster"},\
            default="biggest_inertia"
        Defines how bisection should be performed:

        - "biggest_inertia" means that BisectingKMeans will always check
          all calculated cluster for cluster with biggest SSE
          (Sum of squared errors) and bisect it. This approach concentrates on
          precision, but may be costly in terms of execution time (especially for
          larger amount of data points).

        - "largest_cluster" - BisectingKMeans will always split cluster with
          largest amount of points assigned to it from all clusters
          previously calculated. That should work faster than picking by SSE
          ('biggest_inertia') and may produce similar results in most cases.

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers. If the algorithm stops before fully
        converging (see ``tol`` and ``max_iter``), these will not be
        consistent with ``labels_``.

    labels_ : ndarray of shape (n_samples,)
        Labels of each point.

    inertia_ : float
        Sum of squared distances of samples to their closest cluster center,
        weighted by the sample weights if provided.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    See Also
    --------
    KMeans : Original implementation of K-Means algorithm.

    Notes
    -----
    It might be inefficient when n_cluster is less than 3, due to unnecessary
    calculations for that case.

    Examples
    --------
    >>> from sklearn.cluster import BisectingKMeans
    >>> import numpy as np
    >>> X = np.array([[1, 1], [10, 1], [3, 1],
    ...               [10, 0], [2, 1], [10, 2],
    ...               [10, 8], [10, 9], [10, 10]])
    >>> bisect_means = BisectingKMeans(n_clusters=3, random_state=0).fit(X)
    >>> bisect_means.labels_
    array([0, 2, 0, 2, 0, 2, 1, 1, 1], dtype=int32)
    >>> bisect_means.predict([[0, 0], [12, 3]])
    array([0, 2], dtype=int32)
    >>> bisect_means.cluster_centers_
    array([[ 2., 1.],
           [10., 9.],
           [10., 1.]])

    For a comparison between BisectingKMeans and K-Means refer to example
    :ref:`sphx_glr_auto_examples_cluster_plot_bisect_kmeans.py`.
    """
    _parameter_constraints: dict = ...
    def __init__(self, n_clusters=..., *, init=..., n_init=..., random_state=..., max_iter=..., verbose=..., tol=..., copy_x=..., algorithm=..., bisecting_strategy=...) -> None:
        ...

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=..., sample_weight=...): # -> Self:
        """Compute bisecting k-means clustering.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)

            Training instances to cluster.

            .. note:: The data will be converted to C ordering,
                which will cause a memory copy
                if the given data is not C-contiguous.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight. `sample_weight` is not used during
            initialization if `init` is a callable.

        Returns
        -------
        self
            Fitted estimator.
        """
        ...

    def predict(self, X): # -> _Array[tuple[int], signedinteger[_32Bit]]:
        """Predict which cluster each sample in X belongs to.

        Prediction is made by going down the hierarchical tree
        in searching of closest leaf cluster.

        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        ...

    def __sklearn_tags__(self):
        ...
