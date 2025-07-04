"""
This type stub file was generated by pyright.
"""

from ..base import BaseEstimator, ClassNamePrefixFeaturesOutMixin, TransformerMixin, _fit_context

"""Truncated SVD for sparse matrices, aka latent semantic analysis (LSA)."""
__all__ = ["TruncatedSVD"]
class TruncatedSVD(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    """Dimensionality reduction using truncated SVD (aka LSA).

    This transformer performs linear dimensionality reduction by means of
    truncated singular value decomposition (SVD). Contrary to PCA, this
    estimator does not center the data before computing the singular value
    decomposition. This means it can work with sparse matrices
    efficiently.

    In particular, truncated SVD works on term count/tf-idf matrices as
    returned by the vectorizers in :mod:`sklearn.feature_extraction.text`. In
    that context, it is known as latent semantic analysis (LSA).

    This estimator supports two algorithms: a fast randomized SVD solver, and
    a "naive" algorithm that uses ARPACK as an eigensolver on `X * X.T` or
    `X.T * X`, whichever is more efficient.

    Read more in the :ref:`User Guide <LSA>`.

    Parameters
    ----------
    n_components : int, default=2
        Desired dimensionality of output data.
        If algorithm='arpack', must be strictly less than the number of features.
        If algorithm='randomized', must be less than or equal to the number of features.
        The default value is useful for visualisation. For LSA, a value of
        100 is recommended.

    algorithm : {'arpack', 'randomized'}, default='randomized'
        SVD solver to use. Either "arpack" for the ARPACK wrapper in SciPy
        (scipy.sparse.linalg.svds), or "randomized" for the randomized
        algorithm due to Halko (2009).

    n_iter : int, default=5
        Number of iterations for randomized SVD solver. Not used by ARPACK. The
        default is larger than the default in
        :func:`~sklearn.utils.extmath.randomized_svd` to handle sparse
        matrices that may have large slowly decaying spectrum.

    n_oversamples : int, default=10
        Number of oversamples for randomized SVD solver. Not used by ARPACK.
        See :func:`~sklearn.utils.extmath.randomized_svd` for a complete
        description.

        .. versionadded:: 1.1

    power_iteration_normalizer : {'auto', 'QR', 'LU', 'none'}, default='auto'
        Power iteration normalizer for randomized SVD solver.
        Not used by ARPACK. See :func:`~sklearn.utils.extmath.randomized_svd`
        for more details.

        .. versionadded:: 1.1

    random_state : int, RandomState instance or None, default=None
        Used during randomized svd. Pass an int for reproducible results across
        multiple function calls.
        See :term:`Glossary <random_state>`.

    tol : float, default=0.0
        Tolerance for ARPACK. 0 means machine precision. Ignored by randomized
        SVD solver.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        The right singular vectors of the input data.

    explained_variance_ : ndarray of shape (n_components,)
        The variance of the training samples transformed by a projection to
        each component.

    explained_variance_ratio_ : ndarray of shape (n_components,)
        Percentage of variance explained by each of the selected components.

    singular_values_ : ndarray of shape (n_components,)
        The singular values corresponding to each of the selected components.
        The singular values are equal to the 2-norms of the ``n_components``
        variables in the lower-dimensional space.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    DictionaryLearning : Find a dictionary that sparsely encodes data.
    FactorAnalysis : A simple linear generative model with
        Gaussian latent variables.
    IncrementalPCA : Incremental principal components analysis.
    KernelPCA : Kernel Principal component analysis.
    NMF : Non-Negative Matrix Factorization.
    PCA : Principal component analysis.

    Notes
    -----
    SVD suffers from a problem called "sign indeterminacy", which means the
    sign of the ``components_`` and the output from transform depend on the
    algorithm and random state. To work around this, fit instances of this
    class to data once, then keep the instance around to do transformations.

    References
    ----------
    :arxiv:`Halko, et al. (2009). "Finding structure with randomness:
    Stochastic algorithms for constructing approximate matrix decompositions"
    <0909.4061>`

    Examples
    --------
    >>> from sklearn.decomposition import TruncatedSVD
    >>> from scipy.sparse import csr_matrix
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> X_dense = np.random.rand(100, 100)
    >>> X_dense[:, 2 * np.arange(50)] = 0
    >>> X = csr_matrix(X_dense)
    >>> svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
    >>> svd.fit(X)
    TruncatedSVD(n_components=5, n_iter=7, random_state=42)
    >>> print(svd.explained_variance_ratio_)
    [0.0157 0.0512 0.0499 0.0479 0.0453]
    >>> print(svd.explained_variance_ratio_.sum())
    0.2102
    >>> print(svd.singular_values_)
    [35.2410  4.5981   4.5420  4.4486  4.3288]
    """
    _parameter_constraints: dict = ...
    def __init__(self, n_components=..., *, algorithm=..., n_iter=..., n_oversamples=..., power_iteration_normalizer=..., random_state=..., tol=...) -> None:
        ...

    def fit(self, X, y=...): # -> Self:
        """Fit model on training data X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Returns the transformer object.
        """
        ...

    @_fit_context(prefer_skip_nested_validation=True)
    def fit_transform(self, X, y=...): # -> Any | ndarray[_AnyShape, dtype[float64]] | ndarray[_AnyShape, dtype[Any]]:
        """Fit model to X and perform dimensionality reduction on X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Reduced version of X. This will always be a dense array.
        """
        ...

    def transform(self, X): # -> Any:
        """Perform dimensionality reduction on X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Reduced version of X. This will always be a dense array.
        """
        ...

    def inverse_transform(self, X): # -> Any:
        """Transform X back to its original space.

        Returns an array X_original whose transform would be X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)
            New data.

        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)
            Note that this is always a dense array.
        """
        ...

    def __sklearn_tags__(self):
        ...
