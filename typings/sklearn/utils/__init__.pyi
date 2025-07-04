"""
This type stub file was generated by pyright.
"""

from ..exceptions import DataConversionWarning
from . import metadata_routing
from ._bunch import Bunch
from ._chunking import gen_batches, gen_even_slices
from ._indexing import _safe_indexing, resample, shuffle
from ._mask import safe_mask
from ._repr_html.base import _HTMLDocumentationLinkMixin
from ._repr_html.estimator import estimator_html_repr
from ._tags import ClassifierTags, InputTags, RegressorTags, Tags, TargetTags, TransformerTags, get_tags
from .class_weight import compute_class_weight, compute_sample_weight
from .deprecation import deprecated
from .discovery import all_estimators
from .extmath import safe_sqr
from .murmurhash import murmurhash3_32
from .validation import as_float_array, assert_all_finite, check_X_y, check_array, check_consistent_length, check_random_state, check_scalar, check_symmetric, column_or_1d, indexable

"""Various utilities to help with development."""
__all__ = ["Bunch", "ClassifierTags", "DataConversionWarning", "InputTags", "RegressorTags", "Tags", "TargetTags", "TransformerTags", "all_estimators", "as_float_array", "assert_all_finite", "check_X_y", "check_array", "check_consistent_length", "check_random_state", "check_scalar", "check_symmetric", "column_or_1d", "compute_class_weight", "compute_sample_weight", "deprecated", "estimator_html_repr", "gen_batches", "gen_even_slices", "get_tags", "indexable", "metadata_routing", "murmurhash3_32", "resample", "safe_mask", "safe_sqr", "shuffle"]
