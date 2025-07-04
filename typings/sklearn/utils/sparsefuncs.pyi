"""
This type stub file was generated by pyright.
"""

"""A collection of utilities to work with sparse matrices and arrays."""
def inplace_csr_column_scale(X, scale): # -> None:
    """Inplace column scaling of a CSR matrix.

    Scale each feature of the data matrix by multiplying with specific scale
    provided by the caller assuming a (n_samples, n_features) shape.

    Parameters
    ----------
    X : sparse matrix of shape (n_samples, n_features)
        Matrix to normalize using the variance of the features.
        It should be of CSR format.

    scale : ndarray of shape (n_features,), dtype={np.float32, np.float64}
        Array of precomputed feature-wise values to use for scaling.

    Examples
    --------
    >>> from sklearn.utils import sparsefuncs
    >>> from scipy import sparse
    >>> import numpy as np
    >>> indptr = np.array([0, 3, 4, 4, 4])
    >>> indices = np.array([0, 1, 2, 2])
    >>> data = np.array([8, 1, 2, 5])
    >>> scale = np.array([2, 3, 2])
    >>> csr = sparse.csr_matrix((data, indices, indptr))
    >>> csr.todense()
    matrix([[8, 1, 2],
            [0, 0, 5],
            [0, 0, 0],
            [0, 0, 0]])
    >>> sparsefuncs.inplace_csr_column_scale(csr, scale)
    >>> csr.todense()
    matrix([[16,  3,  4],
            [ 0,  0, 10],
            [ 0,  0,  0],
            [ 0,  0,  0]])
    """
    ...

def inplace_csr_row_scale(X, scale): # -> None:
    """Inplace row scaling of a CSR matrix.

    Scale each sample of the data matrix by multiplying with specific scale
    provided by the caller assuming a (n_samples, n_features) shape.

    Parameters
    ----------
    X : sparse matrix of shape (n_samples, n_features)
        Matrix to be scaled. It should be of CSR format.

    scale : ndarray of float of shape (n_samples,)
        Array of precomputed sample-wise values to use for scaling.
    """
    ...

def mean_variance_axis(X, axis, weights=..., return_sum_weights=...): # -> None:
    """Compute mean and variance along an axis on a CSR or CSC matrix.

    Parameters
    ----------
    X : sparse matrix of shape (n_samples, n_features)
        Input data. It can be of CSR or CSC format.

    axis : {0, 1}
        Axis along which the axis should be computed.

    weights : ndarray of shape (n_samples,) or (n_features,), default=None
        If axis is set to 0 shape is (n_samples,) or
        if axis is set to 1 shape is (n_features,).
        If it is set to None, then samples are equally weighted.

        .. versionadded:: 0.24

    return_sum_weights : bool, default=False
        If True, returns the sum of weights seen for each feature
        if `axis=0` or each sample if `axis=1`.

        .. versionadded:: 0.24

    Returns
    -------

    means : ndarray of shape (n_features,), dtype=floating
        Feature-wise means.

    variances : ndarray of shape (n_features,), dtype=floating
        Feature-wise variances.

    sum_weights : ndarray of shape (n_features,), dtype=floating
        Returned if `return_sum_weights` is `True`.

    Examples
    --------
    >>> from sklearn.utils import sparsefuncs
    >>> from scipy import sparse
    >>> import numpy as np
    >>> indptr = np.array([0, 3, 4, 4, 4])
    >>> indices = np.array([0, 1, 2, 2])
    >>> data = np.array([8, 1, 2, 5])
    >>> scale = np.array([2, 3, 2])
    >>> csr = sparse.csr_matrix((data, indices, indptr))
    >>> csr.todense()
    matrix([[8, 1, 2],
            [0, 0, 5],
            [0, 0, 0],
            [0, 0, 0]])
    >>> sparsefuncs.mean_variance_axis(csr, axis=0)
    (array([2.  , 0.25, 1.75]), array([12.    ,  0.1875,  4.1875]))
    """
    ...

def incr_mean_variance_axis(X, *, axis, last_mean, last_var, last_n, weights=...):
    """Compute incremental mean and variance along an axis on a CSR or CSC matrix.

    last_mean, last_var are the statistics computed at the last step by this
    function. Both must be initialized to 0-arrays of the proper size, i.e.
    the number of features in X. last_n is the number of samples encountered
    until now.

    Parameters
    ----------
    X : CSR or CSC sparse matrix of shape (n_samples, n_features)
        Input data.

    axis : {0, 1}
        Axis along which the axis should be computed.

    last_mean : ndarray of shape (n_features,) or (n_samples,), dtype=floating
        Array of means to update with the new data X.
        Should be of shape (n_features,) if axis=0 or (n_samples,) if axis=1.

    last_var : ndarray of shape (n_features,) or (n_samples,), dtype=floating
        Array of variances to update with the new data X.
        Should be of shape (n_features,) if axis=0 or (n_samples,) if axis=1.

    last_n : float or ndarray of shape (n_features,) or (n_samples,), \
            dtype=floating
        Sum of the weights seen so far, excluding the current weights
        If not float, it should be of shape (n_features,) if
        axis=0 or (n_samples,) if axis=1. If float it corresponds to
        having same weights for all samples (or features).

    weights : ndarray of shape (n_samples,) or (n_features,), default=None
        If axis is set to 0 shape is (n_samples,) or
        if axis is set to 1 shape is (n_features,).
        If it is set to None, then samples are equally weighted.

        .. versionadded:: 0.24

    Returns
    -------
    means : ndarray of shape (n_features,) or (n_samples,), dtype=floating
        Updated feature-wise means if axis = 0 or
        sample-wise means if axis = 1.

    variances : ndarray of shape (n_features,) or (n_samples,), dtype=floating
        Updated feature-wise variances if axis = 0 or
        sample-wise variances if axis = 1.

    n : ndarray of shape (n_features,) or (n_samples,), dtype=integral
        Updated number of seen samples per feature if axis=0
        or number of seen features per sample if axis=1.

        If weights is not None, n is a sum of the weights of the seen
        samples or features instead of the actual number of seen
        samples or features.

    Notes
    -----
    NaNs are ignored in the algorithm.

    Examples
    --------
    >>> from sklearn.utils import sparsefuncs
    >>> from scipy import sparse
    >>> import numpy as np
    >>> indptr = np.array([0, 3, 4, 4, 4])
    >>> indices = np.array([0, 1, 2, 2])
    >>> data = np.array([8, 1, 2, 5])
    >>> scale = np.array([2, 3, 2])
    >>> csr = sparse.csr_matrix((data, indices, indptr))
    >>> csr.todense()
    matrix([[8, 1, 2],
            [0, 0, 5],
            [0, 0, 0],
            [0, 0, 0]])
    >>> sparsefuncs.incr_mean_variance_axis(
    ...     csr, axis=0, last_mean=np.zeros(3), last_var=np.zeros(3), last_n=2
    ... )
    (array([1.33, 0.167, 1.17]), array([8.88, 0.139, 3.47]),
    array([6., 6., 6.]))
    """
    ...

def inplace_column_scale(X, scale): # -> None:
    """Inplace column scaling of a CSC/CSR matrix.

    Scale each feature of the data matrix by multiplying with specific scale
    provided by the caller assuming a (n_samples, n_features) shape.

    Parameters
    ----------
    X : sparse matrix of shape (n_samples, n_features)
        Matrix to normalize using the variance of the features. It should be
        of CSC or CSR format.

    scale : ndarray of shape (n_features,), dtype={np.float32, np.float64}
        Array of precomputed feature-wise values to use for scaling.

    Examples
    --------
    >>> from sklearn.utils import sparsefuncs
    >>> from scipy import sparse
    >>> import numpy as np
    >>> indptr = np.array([0, 3, 4, 4, 4])
    >>> indices = np.array([0, 1, 2, 2])
    >>> data = np.array([8, 1, 2, 5])
    >>> scale = np.array([2, 3, 2])
    >>> csr = sparse.csr_matrix((data, indices, indptr))
    >>> csr.todense()
    matrix([[8, 1, 2],
            [0, 0, 5],
            [0, 0, 0],
            [0, 0, 0]])
    >>> sparsefuncs.inplace_column_scale(csr, scale)
    >>> csr.todense()
    matrix([[16,  3,  4],
            [ 0,  0, 10],
            [ 0,  0,  0],
            [ 0,  0,  0]])
    """
    ...

def inplace_row_scale(X, scale): # -> None:
    """Inplace row scaling of a CSR or CSC matrix.

    Scale each row of the data matrix by multiplying with specific scale
    provided by the caller assuming a (n_samples, n_features) shape.

    Parameters
    ----------
    X : sparse matrix of shape (n_samples, n_features)
        Matrix to be scaled. It should be of CSR or CSC format.

    scale : ndarray of shape (n_features,), dtype={np.float32, np.float64}
        Array of precomputed sample-wise values to use for scaling.

    Examples
    --------
    >>> from sklearn.utils import sparsefuncs
    >>> from scipy import sparse
    >>> import numpy as np
    >>> indptr = np.array([0, 2, 3, 4, 5])
    >>> indices = np.array([0, 1, 2, 3, 3])
    >>> data = np.array([8, 1, 2, 5, 6])
    >>> scale = np.array([2, 3, 4, 5])
    >>> csr = sparse.csr_matrix((data, indices, indptr))
    >>> csr.todense()
    matrix([[8, 1, 0, 0],
            [0, 0, 2, 0],
            [0, 0, 0, 5],
            [0, 0, 0, 6]])
    >>> sparsefuncs.inplace_row_scale(csr, scale)
    >>> csr.todense()
     matrix([[16,  2,  0,  0],
             [ 0,  0,  6,  0],
             [ 0,  0,  0, 20],
             [ 0,  0,  0, 30]])
    """
    ...

def inplace_swap_row_csc(X, m, n): # -> None:
    """Swap two rows of a CSC matrix in-place.

    Parameters
    ----------
    X : sparse matrix of shape (n_samples, n_features)
        Matrix whose two rows are to be swapped. It should be of
        CSC format.

    m : int
        Index of the row of X to be swapped.

    n : int
        Index of the row of X to be swapped.
    """
    ...

def inplace_swap_row_csr(X, m, n): # -> None:
    """Swap two rows of a CSR matrix in-place.

    Parameters
    ----------
    X : sparse matrix of shape (n_samples, n_features)
        Matrix whose two rows are to be swapped. It should be of
        CSR format.

    m : int
        Index of the row of X to be swapped.

    n : int
        Index of the row of X to be swapped.
    """
    ...

def inplace_swap_row(X, m, n): # -> None:
    """
    Swap two rows of a CSC/CSR matrix in-place.

    Parameters
    ----------
    X : sparse matrix of shape (n_samples, n_features)
        Matrix whose two rows are to be swapped. It should be of CSR or
        CSC format.

    m : int
        Index of the row of X to be swapped.

    n : int
        Index of the row of X to be swapped.

    Examples
    --------
    >>> from sklearn.utils import sparsefuncs
    >>> from scipy import sparse
    >>> import numpy as np
    >>> indptr = np.array([0, 2, 3, 3, 3])
    >>> indices = np.array([0, 2, 2])
    >>> data = np.array([8, 2, 5])
    >>> csr = sparse.csr_matrix((data, indices, indptr))
    >>> csr.todense()
    matrix([[8, 0, 2],
            [0, 0, 5],
            [0, 0, 0],
            [0, 0, 0]])
    >>> sparsefuncs.inplace_swap_row(csr, 0, 1)
    >>> csr.todense()
    matrix([[0, 0, 5],
            [8, 0, 2],
            [0, 0, 0],
            [0, 0, 0]])
    """
    ...

def inplace_swap_column(X, m, n): # -> None:
    """
    Swap two columns of a CSC/CSR matrix in-place.

    Parameters
    ----------
    X : sparse matrix of shape (n_samples, n_features)
        Matrix whose two columns are to be swapped. It should be of
        CSR or CSC format.

    m : int
        Index of the column of X to be swapped.

    n : int
        Index of the column of X to be swapped.

    Examples
    --------
    >>> from sklearn.utils import sparsefuncs
    >>> from scipy import sparse
    >>> import numpy as np
    >>> indptr = np.array([0, 2, 3, 3, 3])
    >>> indices = np.array([0, 2, 2])
    >>> data = np.array([8, 2, 5])
    >>> csr = sparse.csr_matrix((data, indices, indptr))
    >>> csr.todense()
    matrix([[8, 0, 2],
            [0, 0, 5],
            [0, 0, 0],
            [0, 0, 0]])
    >>> sparsefuncs.inplace_swap_column(csr, 0, 1)
    >>> csr.todense()
    matrix([[0, 8, 2],
            [0, 0, 5],
            [0, 0, 0],
            [0, 0, 0]])
    """
    ...

def min_max_axis(X, axis, ignore_nan=...): # -> tuple[Any, Any] | None:
    """Compute minimum and maximum along an axis on a CSR or CSC matrix.

     Optionally ignore NaN values.

    Parameters
    ----------
    X : sparse matrix of shape (n_samples, n_features)
        Input data. It should be of CSR or CSC format.

    axis : {0, 1}
        Axis along which the axis should be computed.

    ignore_nan : bool, default=False
        Ignore or passing through NaN values.

        .. versionadded:: 0.20

    Returns
    -------

    mins : ndarray of shape (n_features,), dtype={np.float32, np.float64}
        Feature-wise minima.

    maxs : ndarray of shape (n_features,), dtype={np.float32, np.float64}
        Feature-wise maxima.
    """
    ...

def count_nonzero(X, axis=..., sample_weight=...): # -> Any | ndarray[_AnyShape, dtype[Any]] | NDArray[intp]:
    """A variant of X.getnnz() with extension to weighting on axis 0.

    Useful in efficiently calculating multilabel metrics.

    Parameters
    ----------
    X : sparse matrix of shape (n_samples, n_labels)
        Input data. It should be of CSR format.

    axis : {0, 1}, default=None
        The axis on which the data is aggregated.

    sample_weight : array-like of shape (n_samples,), default=None
        Weight for each row of X.

    Returns
    -------
    nnz : int, float, ndarray of shape (n_samples,) or ndarray of shape (n_features,)
        Number of non-zero values in the array along a given axis. Otherwise,
        the total number of non-zero values in the array is returned.
    """
    ...

def csc_median_axis_0(X): # -> _Array1D[float64]:
    """Find the median across axis 0 of a CSC matrix.

    It is equivalent to doing np.median(X, axis=0).

    Parameters
    ----------
    X : sparse matrix of shape (n_samples, n_features)
        Input data. It should be of CSC format.

    Returns
    -------
    median : ndarray of shape (n_features,)
        Median.
    """
    ...
