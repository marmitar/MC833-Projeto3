"""
This type stub file was generated by pyright.
"""

from numbers import Integral
from ..base import BaseEstimator, ClassNamePrefixFeaturesOutMixin, ClusterMixin, _fit_context
from ..utils._param_validation import Interval, validate_params
from ._feature_agglomeration import AgglomerationTransform

"""Hierarchical Agglomerative Clustering

These routines perform some hierarchical agglomerative clustering of some
input data.
"""
@validate_params({ "X": ["array-like"],"connectivity": ["array-like", "sparse matrix", None],"n_clusters": [Interval(Integral, 1, None, closed="left"), None],"return_distance": ["boolean"] }, prefer_skip_nested_validation=True)
def ward_tree(X, *, connectivity=..., n_clusters=..., return_distance=...): # -> tuple[Any, Literal[1], int | Any, None, Any] | tuple[Any, Literal[1], int | Any, None] | tuple[NDArray[Any], Any, int | Any, _Array1D[signedinteger[_NBitIntP]], NDArray[Any]] | tuple[NDArray[Any], Any, int | Any, _Array1D[signedinteger[_NBitIntP]]]:
    """Ward clustering based on a Feature matrix.

    Recursively merges the pair of clusters that minimally increases
    within-cluster variance.

    The inertia matrix uses a Heapq-based representation.

    This is the structured version, that takes into account some topological
    structure between samples.

    Read more in the :ref:`User Guide <hierarchical_clustering>`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix representing `n_samples` samples to be clustered.

    connectivity : {array-like, sparse matrix}, default=None
        Connectivity matrix. Defines for each sample the neighboring samples
        following a given structure of the data. The matrix is assumed to
        be symmetric and only the upper triangular half is used.
        Default is None, i.e, the Ward algorithm is unstructured.

    n_clusters : int, default=None
        `n_clusters` should be less than `n_samples`.  Stop early the
        construction of the tree at `n_clusters.` This is useful to decrease
        computation time if the number of clusters is not small compared to the
        number of samples. In this case, the complete tree is not computed, thus
        the 'children' output is of limited use, and the 'parents' output should
        rather be used. This option is valid only when specifying a connectivity
        matrix.

    return_distance : bool, default=False
        If `True`, return the distance between the clusters.

    Returns
    -------
    children : ndarray of shape (n_nodes-1, 2)
        The children of each non-leaf node. Values less than `n_samples`
        correspond to leaves of the tree which are the original samples.
        A node `i` greater than or equal to `n_samples` is a non-leaf
        node and has children `children_[i - n_samples]`. Alternatively
        at the i-th iteration, children[i][0] and children[i][1]
        are merged to form node `n_samples + i`.

    n_connected_components : int
        The number of connected components in the graph.

    n_leaves : int
        The number of leaves in the tree.

    parents : ndarray of shape (n_nodes,) or None
        The parent of each node. Only returned when a connectivity matrix
        is specified, elsewhere 'None' is returned.

    distances : ndarray of shape (n_nodes-1,)
        Only returned if `return_distance` is set to `True` (for compatibility).
        The distances between the centers of the nodes. `distances[i]`
        corresponds to a weighted Euclidean distance between
        the nodes `children[i, 1]` and `children[i, 2]`. If the nodes refer to
        leaves of the tree, then `distances[i]` is their unweighted Euclidean
        distance. Distances are updated in the following way
        (from scipy.hierarchy.linkage):

        The new entry :math:`d(u,v)` is computed as follows,

        .. math::

           d(u,v) = \\sqrt{\\frac{|v|+|s|}
                               {T}d(v,s)^2
                        + \\frac{|v|+|t|}
                               {T}d(v,t)^2
                        - \\frac{|v|}
                               {T}d(s,t)^2}

        where :math:`u` is the newly joined cluster consisting of
        clusters :math:`s` and :math:`t`, :math:`v` is an unused
        cluster in the forest, :math:`T=|v|+|s|+|t|`, and
        :math:`|*|` is the cardinality of its argument. This is also
        known as the incremental algorithm.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.cluster import ward_tree
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [4, 2], [4, 4], [4, 0]])
    >>> children, n_connected_components, n_leaves, parents = ward_tree(X)
    >>> children
    array([[0, 1],
           [3, 5],
           [2, 6],
           [4, 7],
           [8, 9]])
    >>> n_connected_components
    1
    >>> n_leaves
    6
    """
    ...

def linkage_tree(X, connectivity=..., n_clusters=..., linkage=..., affinity=..., return_distance=...): # -> tuple[Any, Literal[1], int | Any, None, Any] | tuple[Any, Literal[1], int | Any, None] | tuple[Any, Any, int | Any, _Array1D[signedinteger[_NBitIntP]], Any] | tuple[Any, Any, int | Any, _Array1D[signedinteger[_NBitIntP]]] | tuple[ndarray[_AnyShape, dtype[Any]], Any, int | Any, _Array1D[signedinteger[_NBitIntP]], _Array1D[float64] | ndarray[_AnyShape, dtype[float64]] | Any | None] | tuple[ndarray[_AnyShape, dtype[Any]], Any, int | Any, _Array1D[signedinteger[_NBitIntP]]]:
    """Linkage agglomerative clustering based on a Feature matrix.

    The inertia matrix uses a Heapq-based representation.

    This is the structured version, that takes into account some topological
    structure between samples.

    Read more in the :ref:`User Guide <hierarchical_clustering>`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix representing `n_samples` samples to be clustered.

    connectivity : sparse matrix, default=None
        Connectivity matrix. Defines for each sample the neighboring samples
        following a given structure of the data. The matrix is assumed to
        be symmetric and only the upper triangular half is used.
        Default is `None`, i.e, the Ward algorithm is unstructured.

    n_clusters : int, default=None
        Stop early the construction of the tree at `n_clusters`. This is
        useful to decrease computation time if the number of clusters is
        not small compared to the number of samples. In this case, the
        complete tree is not computed, thus the 'children' output is of
        limited use, and the 'parents' output should rather be used.
        This option is valid only when specifying a connectivity matrix.

    linkage : {"average", "complete", "single"}, default="complete"
        Which linkage criteria to use. The linkage criterion determines which
        distance to use between sets of observation.
            - "average" uses the average of the distances of each observation of
              the two sets.
            - "complete" or maximum linkage uses the maximum distances between
              all observations of the two sets.
            - "single" uses the minimum of the distances between all
              observations of the two sets.

    affinity : str or callable, default='euclidean'
        Which metric to use. Can be 'euclidean', 'manhattan', or any
        distance known to paired distance (see metric.pairwise).

    return_distance : bool, default=False
        Whether or not to return the distances between the clusters.

    Returns
    -------
    children : ndarray of shape (n_nodes-1, 2)
        The children of each non-leaf node. Values less than `n_samples`
        correspond to leaves of the tree which are the original samples.
        A node `i` greater than or equal to `n_samples` is a non-leaf
        node and has children `children_[i - n_samples]`. Alternatively
        at the i-th iteration, children[i][0] and children[i][1]
        are merged to form node `n_samples + i`.

    n_connected_components : int
        The number of connected components in the graph.

    n_leaves : int
        The number of leaves in the tree.

    parents : ndarray of shape (n_nodes, ) or None
        The parent of each node. Only returned when a connectivity matrix
        is specified, elsewhere 'None' is returned.

    distances : ndarray of shape (n_nodes-1,)
        Returned when `return_distance` is set to `True`.

        distances[i] refers to the distance between children[i][0] and
        children[i][1] when they are merged.

    See Also
    --------
    ward_tree : Hierarchical clustering with ward linkage.
    """
    ...

_TREE_BUILDERS = ...
class AgglomerativeClustering(ClusterMixin, BaseEstimator):
    """
    Agglomerative Clustering.

    Recursively merges pair of clusters of sample data; uses linkage distance.

    Read more in the :ref:`User Guide <hierarchical_clustering>`.

    Parameters
    ----------
    n_clusters : int or None, default=2
        The number of clusters to find. It must be ``None`` if
        ``distance_threshold`` is not ``None``.

    metric : str or callable, default="euclidean"
        Metric used to compute the linkage. Can be "euclidean", "l1", "l2",
        "manhattan", "cosine", or "precomputed". If linkage is "ward", only
        "euclidean" is accepted. If "precomputed", a distance matrix is needed
        as input for the fit method. If connectivity is None, linkage is
        "single" and affinity is not "precomputed" any valid pairwise distance
        metric can be assigned.

        For an example of agglomerative clustering with different metrics, see
        :ref:`sphx_glr_auto_examples_cluster_plot_agglomerative_clustering_metrics.py`.

        .. versionadded:: 1.2

    memory : str or object with the joblib.Memory interface, default=None
        Used to cache the output of the computation of the tree.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    connectivity : array-like, sparse matrix, or callable, default=None
        Connectivity matrix. Defines for each sample the neighboring
        samples following a given structure of the data.
        This can be a connectivity matrix itself or a callable that transforms
        the data into a connectivity matrix, such as derived from
        `kneighbors_graph`. Default is ``None``, i.e, the
        hierarchical clustering algorithm is unstructured.

        For an example of connectivity matrix using
        :class:`~sklearn.neighbors.kneighbors_graph`, see
        :ref:`sphx_glr_auto_examples_cluster_plot_agglomerative_clustering.py`.

    compute_full_tree : 'auto' or bool, default='auto'
        Stop early the construction of the tree at ``n_clusters``. This is
        useful to decrease computation time if the number of clusters is not
        small compared to the number of samples. This option is useful only
        when specifying a connectivity matrix. Note also that when varying the
        number of clusters and using caching, it may be advantageous to compute
        the full tree. It must be ``True`` if ``distance_threshold`` is not
        ``None``. By default `compute_full_tree` is "auto", which is equivalent
        to `True` when `distance_threshold` is not `None` or that `n_clusters`
        is inferior to the maximum between 100 or `0.02 * n_samples`.
        Otherwise, "auto" is equivalent to `False`.

    linkage : {'ward', 'complete', 'average', 'single'}, default='ward'
        Which linkage criterion to use. The linkage criterion determines which
        distance to use between sets of observation. The algorithm will merge
        the pairs of cluster that minimize this criterion.

        - 'ward' minimizes the variance of the clusters being merged.
        - 'average' uses the average of the distances of each observation of
          the two sets.
        - 'complete' or 'maximum' linkage uses the maximum distances between
          all observations of the two sets.
        - 'single' uses the minimum of the distances between all observations
          of the two sets.

        .. versionadded:: 0.20
            Added the 'single' option

        For examples comparing different `linkage` criteria, see
        :ref:`sphx_glr_auto_examples_cluster_plot_linkage_comparison.py`.

    distance_threshold : float, default=None
        The linkage distance threshold at or above which clusters will not be
        merged. If not ``None``, ``n_clusters`` must be ``None`` and
        ``compute_full_tree`` must be ``True``.

        .. versionadded:: 0.21

    compute_distances : bool, default=False
        Computes distances between clusters even if `distance_threshold` is not
        used. This can be used to make dendrogram visualization, but introduces
        a computational and memory overhead.

        .. versionadded:: 0.24

        For an example of dendrogram visualization, see
        :ref:`sphx_glr_auto_examples_cluster_plot_agglomerative_dendrogram.py`.

    Attributes
    ----------
    n_clusters_ : int
        The number of clusters found by the algorithm. If
        ``distance_threshold=None``, it will be equal to the given
        ``n_clusters``.

    labels_ : ndarray of shape (n_samples)
        Cluster labels for each point.

    n_leaves_ : int
        Number of leaves in the hierarchical tree.

    n_connected_components_ : int
        The estimated number of connected components in the graph.

        .. versionadded:: 0.21
            ``n_connected_components_`` was added to replace ``n_components_``.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    children_ : array-like of shape (n_samples-1, 2)
        The children of each non-leaf node. Values less than `n_samples`
        correspond to leaves of the tree which are the original samples.
        A node `i` greater than or equal to `n_samples` is a non-leaf
        node and has children `children_[i - n_samples]`. Alternatively
        at the i-th iteration, children[i][0] and children[i][1]
        are merged to form node `n_samples + i`.

    distances_ : array-like of shape (n_nodes-1,)
        Distances between nodes in the corresponding place in `children_`.
        Only computed if `distance_threshold` is used or `compute_distances`
        is set to `True`.

    See Also
    --------
    FeatureAgglomeration : Agglomerative clustering but for features instead of
        samples.
    ward_tree : Hierarchical clustering with ward linkage.

    Examples
    --------
    >>> from sklearn.cluster import AgglomerativeClustering
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [4, 2], [4, 4], [4, 0]])
    >>> clustering = AgglomerativeClustering().fit(X)
    >>> clustering
    AgglomerativeClustering()
    >>> clustering.labels_
    array([1, 1, 1, 0, 0, 0])

    For a comparison of Agglomerative clustering with other clustering algorithms, see
    :ref:`sphx_glr_auto_examples_cluster_plot_cluster_comparison.py`
    """
    _parameter_constraints: dict = ...
    def __init__(self, n_clusters=..., *, metric=..., memory=..., connectivity=..., compute_full_tree=..., linkage=..., distance_threshold=..., compute_distances=...) -> None:
        ...

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=...): # -> Self:
        """Fit the hierarchical clustering from features, or distance matrix.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features) or \
                (n_samples, n_samples)
            Training instances to cluster, or distances between instances if
            ``metric='precomputed'``.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Returns the fitted instance.
        """
        ...

    def fit_predict(self, X, y=...):
        """Fit and return the result of each sample's clustering assignment.

        In addition to fitting, this method also return the result of the
        clustering assignment for each sample in the training set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or \
                (n_samples, n_samples)
            Training instances to cluster, or distances between instances if
            ``affinity='precomputed'``.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels.
        """
        ...



class FeatureAgglomeration(ClassNamePrefixFeaturesOutMixin, AgglomerationTransform, AgglomerativeClustering):
    """Agglomerate features.

    Recursively merges pair of clusters of features.

    Refer to
    :ref:`sphx_glr_auto_examples_cluster_plot_feature_agglomeration_vs_univariate_selection.py`
    for an example comparison of :class:`FeatureAgglomeration` strategy with a
    univariate feature selection strategy (based on ANOVA).

    Read more in the :ref:`User Guide <hierarchical_clustering>`.

    Parameters
    ----------
    n_clusters : int or None, default=2
        The number of clusters to find. It must be ``None`` if
        ``distance_threshold`` is not ``None``.

    metric : str or callable, default="euclidean"
        Metric used to compute the linkage. Can be "euclidean", "l1", "l2",
        "manhattan", "cosine", or "precomputed". If linkage is "ward", only
        "euclidean" is accepted. If "precomputed", a distance matrix is needed
        as input for the fit method.

        .. versionadded:: 1.2

    memory : str or object with the joblib.Memory interface, default=None
        Used to cache the output of the computation of the tree.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    connectivity : array-like, sparse matrix, or callable, default=None
        Connectivity matrix. Defines for each feature the neighboring
        features following a given structure of the data.
        This can be a connectivity matrix itself or a callable that transforms
        the data into a connectivity matrix, such as derived from
        `kneighbors_graph`. Default is `None`, i.e, the
        hierarchical clustering algorithm is unstructured.

    compute_full_tree : 'auto' or bool, default='auto'
        Stop early the construction of the tree at `n_clusters`. This is useful
        to decrease computation time if the number of clusters is not small
        compared to the number of features. This option is useful only when
        specifying a connectivity matrix. Note also that when varying the
        number of clusters and using caching, it may be advantageous to compute
        the full tree. It must be ``True`` if ``distance_threshold`` is not
        ``None``. By default `compute_full_tree` is "auto", which is equivalent
        to `True` when `distance_threshold` is not `None` or that `n_clusters`
        is inferior to the maximum between 100 or `0.02 * n_samples`.
        Otherwise, "auto" is equivalent to `False`.

    linkage : {"ward", "complete", "average", "single"}, default="ward"
        Which linkage criterion to use. The linkage criterion determines which
        distance to use between sets of features. The algorithm will merge
        the pairs of cluster that minimize this criterion.

        - "ward" minimizes the variance of the clusters being merged.
        - "complete" or maximum linkage uses the maximum distances between
          all features of the two sets.
        - "average" uses the average of the distances of each feature of
          the two sets.
        - "single" uses the minimum of the distances between all features
          of the two sets.

    pooling_func : callable, default=np.mean
        This combines the values of agglomerated features into a single
        value, and should accept an array of shape [M, N] and the keyword
        argument `axis=1`, and reduce it to an array of size [M].

    distance_threshold : float, default=None
        The linkage distance threshold at or above which clusters will not be
        merged. If not ``None``, ``n_clusters`` must be ``None`` and
        ``compute_full_tree`` must be ``True``.

        .. versionadded:: 0.21

    compute_distances : bool, default=False
        Computes distances between clusters even if `distance_threshold` is not
        used. This can be used to make dendrogram visualization, but introduces
        a computational and memory overhead.

        .. versionadded:: 0.24

    Attributes
    ----------
    n_clusters_ : int
        The number of clusters found by the algorithm. If
        ``distance_threshold=None``, it will be equal to the given
        ``n_clusters``.

    labels_ : array-like of (n_features,)
        Cluster labels for each feature.

    n_leaves_ : int
        Number of leaves in the hierarchical tree.

    n_connected_components_ : int
        The estimated number of connected components in the graph.

        .. versionadded:: 0.21
            ``n_connected_components_`` was added to replace ``n_components_``.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    children_ : array-like of shape (n_nodes-1, 2)
        The children of each non-leaf node. Values less than `n_features`
        correspond to leaves of the tree which are the original samples.
        A node `i` greater than or equal to `n_features` is a non-leaf
        node and has children `children_[i - n_features]`. Alternatively
        at the i-th iteration, children[i][0] and children[i][1]
        are merged to form node `n_features + i`.

    distances_ : array-like of shape (n_nodes-1,)
        Distances between nodes in the corresponding place in `children_`.
        Only computed if `distance_threshold` is used or `compute_distances`
        is set to `True`.

    See Also
    --------
    AgglomerativeClustering : Agglomerative clustering samples instead of
        features.
    ward_tree : Hierarchical clustering with ward linkage.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import datasets, cluster
    >>> digits = datasets.load_digits()
    >>> images = digits.images
    >>> X = np.reshape(images, (len(images), -1))
    >>> agglo = cluster.FeatureAgglomeration(n_clusters=32)
    >>> agglo.fit(X)
    FeatureAgglomeration(n_clusters=32)
    >>> X_reduced = agglo.transform(X)
    >>> X_reduced.shape
    (1797, 32)
    """
    _parameter_constraints: dict = ...
    def __init__(self, n_clusters=..., *, metric=..., memory=..., connectivity=..., compute_full_tree=..., linkage=..., pooling_func=..., distance_threshold=..., compute_distances=...) -> None:
        ...

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=...): # -> Self:
        """Fit the hierarchical clustering on the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Returns the transformer.
        """
        ...

    @property
    def fit_predict(self):
        """Fit and return the result of each sample's clustering assignment."""
        ...
