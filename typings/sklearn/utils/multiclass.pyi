"""
This type stub file was generated by pyright.
"""

"""Utilities to handle multiclass/multioutput target in classifiers."""
_FN_UNIQUE_LABELS = ...
def unique_labels(*ys): # -> Any:
    """Extract an ordered array of unique labels.

    We don't allow:
        - mix of multilabel and multiclass (single label) targets
        - mix of label indicator matrix and anything else,
          because there are no explicit labels)
        - mix of label indicator matrices of different sizes
        - mix of string and integer labels

    At the moment, we also don't allow "multiclass-multioutput" input type.

    Parameters
    ----------
    *ys : array-likes
        Label values.

    Returns
    -------
    out : ndarray of shape (n_unique_labels,)
        An ordered array of unique labels.

    Examples
    --------
    >>> from sklearn.utils.multiclass import unique_labels
    >>> unique_labels([3, 5, 5, 5, 7, 7])
    array([3, 5, 7])
    >>> unique_labels([1, 2, 3, 4], [2, 2, 3, 4])
    array([1, 2, 3, 4])
    >>> unique_labels([1, 2, 10], [5, 11])
    array([ 1,  2,  5, 10, 11])
    """
    ...

def is_multilabel(y): # -> Any | bool:
    """Check if ``y`` is in a multilabel format.

    Parameters
    ----------
    y : ndarray of shape (n_samples,)
        Target values.

    Returns
    -------
    out : bool
        Return ``True``, if ``y`` is in a multilabel format, else ``False``.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.utils.multiclass import is_multilabel
    >>> is_multilabel([0, 1, 0, 1])
    False
    >>> is_multilabel([[1], [0, 2], []])
    False
    >>> is_multilabel(np.array([[1, 0], [0, 0]]))
    True
    >>> is_multilabel(np.array([[1], [0], [0]]))
    False
    >>> is_multilabel(np.array([[1, 0, 0]]))
    True
    """
    ...

def check_classification_targets(y): # -> None:
    """Ensure that target y is of a non-regression type.

    Only the following target types (as defined in type_of_target) are allowed:
        'binary', 'multiclass', 'multiclass-multioutput',
        'multilabel-indicator', 'multilabel-sequences'

    Parameters
    ----------
    y : array-like
        Target values.
    """
    ...

def type_of_target(y, input_name=..., raise_unknown=...):
    """Determine the type of data indicated by the target.

    Note that this type is the most specific type that can be inferred.
    For example:

    * ``binary`` is more specific but compatible with ``multiclass``.
    * ``multiclass`` of integers is more specific but compatible with ``continuous``.
    * ``multilabel-indicator`` is more specific but compatible with
      ``multiclass-multioutput``.

    Parameters
    ----------
    y : {array-like, sparse matrix}
        Target values. If a sparse matrix, `y` is expected to be a
        CSR/CSC matrix.

    input_name : str, default=""
        The data name used to construct the error message.

        .. versionadded:: 1.1.0

    raise_unknown : bool, default=False
        If `True`, raise an error when the type of target returned by
        :func:`~sklearn.utils.multiclass.type_of_target` is `"unknown"`.

        .. versionadded:: 1.6

    Returns
    -------
    target_type : str
        One of:

        * 'continuous': `y` is an array-like of floats that are not all
          integers, and is 1d or a column vector.
        * 'continuous-multioutput': `y` is a 2d array of floats that are
          not all integers, and both dimensions are of size > 1.
        * 'binary': `y` contains <= 2 discrete values and is 1d or a column
          vector.
        * 'multiclass': `y` contains more than two discrete values, is not a
          sequence of sequences, and is 1d or a column vector.
        * 'multiclass-multioutput': `y` is a 2d array that contains more
          than two discrete values, is not a sequence of sequences, and both
          dimensions are of size > 1.
        * 'multilabel-indicator': `y` is a label indicator matrix, an array
          of two dimensions with at least two columns, and at most 2 unique
          values.
        * 'unknown': `y` is array-like but none of the above, such as a 3d
          array, sequence of sequences, or an array of non-sequence objects.

    Examples
    --------
    >>> from sklearn.utils.multiclass import type_of_target
    >>> import numpy as np
    >>> type_of_target([0.1, 0.6])
    'continuous'
    >>> type_of_target([1, -1, -1, 1])
    'binary'
    >>> type_of_target(['a', 'b', 'a'])
    'binary'
    >>> type_of_target([1.0, 2.0])
    'binary'
    >>> type_of_target([1, 0, 2])
    'multiclass'
    >>> type_of_target([1.0, 0.0, 3.0])
    'multiclass'
    >>> type_of_target(['a', 'b', 'c'])
    'multiclass'
    >>> type_of_target(np.array([[1, 2], [3, 1]]))
    'multiclass-multioutput'
    >>> type_of_target([[1, 2]])
    'multilabel-indicator'
    >>> type_of_target(np.array([[1.5, 2.0], [3.0, 1.6]]))
    'continuous-multioutput'
    >>> type_of_target(np.array([[0, 1], [1, 1]]))
    'multilabel-indicator'
    """
    ...

def class_distribution(y, sample_weight=...): # -> tuple[list[Any], list[Any], list[Any]]:
    """Compute class priors from multioutput-multiclass target data.

    Parameters
    ----------
    y : {array-like, sparse matrix} of size (n_samples, n_outputs)
        The labels for each example.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    classes : list of size n_outputs of ndarray of size (n_classes,)
        List of classes for each column.

    n_classes : list of int of size n_outputs
        Number of classes in each column.

    class_prior : list of size n_outputs of ndarray of size (n_classes,)
        Class distribution of each column.
    """
    ...
