"""
This type stub file was generated by pyright.
"""

from ._param_validation import validate_params

@validate_params({ "X": ["array-like", "sparse matrix"],"mask": ["array-like"] }, prefer_skip_nested_validation=True)
def safe_mask(X, mask): # -> NDArray[Any]:
    """Return a mask which is safe to use on X.

    Parameters
    ----------
    X : {array-like, sparse matrix}
        Data on which to apply mask.

    mask : array-like
        Mask to be used on X.

    Returns
    -------
    mask : ndarray
        Array that is safe to use on X.

    Examples
    --------
    >>> from sklearn.utils import safe_mask
    >>> from scipy.sparse import csr_matrix
    >>> data = csr_matrix([[1], [2], [3], [4], [5]])
    >>> condition = [False, True, True, False, True]
    >>> mask = safe_mask(data, condition)
    >>> data[mask].toarray()
    array([[2],
           [3],
           [5]])
    """
    ...

def axis0_safe_slice(X, mask, len_mask): # -> _Array[tuple[int, int], float64]:
    """Return a mask which is safer to use on X than safe_mask.

    This mask is safer than safe_mask since it returns an
    empty array, when a sparse matrix is sliced with a boolean mask
    with all False, instead of raising an unhelpful error in older
    versions of SciPy.

    See: https://github.com/scipy/scipy/issues/5361

    Also note that we can avoid doing the dot product by checking if
    the len_mask is not zero in _huber_loss_and_gradient but this
    is not going to be the bottleneck, since the number of outliers
    and non_outliers are typically non-zero and it makes the code
    tougher to follow.

    Parameters
    ----------
    X : {array-like, sparse matrix}
        Data on which to apply mask.

    mask : ndarray
        Mask to be used on X.

    len_mask : int
        The length of the mask.

    Returns
    -------
    mask : ndarray
        Array that is safe to use on X.
    """
    ...

def indices_to_mask(indices, mask_length): # -> _Array1D[Any]:
    """Convert list of indices to boolean mask.

    Parameters
    ----------
    indices : list-like
        List of integers treated as indices.
    mask_length : int
        Length of boolean mask to be generated.
        This parameter must be greater than max(indices).

    Returns
    -------
    mask : 1d boolean nd-array
        Boolean array that is True where indices are present, else False.

    Examples
    --------
    >>> from sklearn.utils._mask import indices_to_mask
    >>> indices = [1, 2 , 3, 4]
    >>> indices_to_mask(indices, 5)
    array([False,  True,  True,  True,  True])
    """
    ...
