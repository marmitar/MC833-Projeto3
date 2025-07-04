"""
This type stub file was generated by pyright.
"""

from statsmodels.tools.tools import Bunch

"""assert functions from numpy and pandas testing

"""
PARAM_LIST = ...
def bunch_factory(attribute, columns): # -> type[FactoryBunch]:
    """
    Generates a special purpose Bunch class

    Parameters
    ----------
    attribute: str
        Attribute to access when splitting
    columns: List[str]
        List of names to use when splitting the columns of attribute

    Notes
    -----
    After the class is initialized as a Bunch, the columne of attribute
    are split so that Bunch has the keys in columns and
    bunch[column[i]] = bunch[attribute][:, i]
    """
    class FactoryBunch(Bunch):
        ...



ParamsTableTestBunch = ...
MarginTableTestBunch = ...
class Holder:
    """
    Test-focused class to simplify accessing values by attribute
    """
    def __init__(self, **kwds) -> None:
        ...

    def __str__(self) -> str:
        ...

    def __repr__(self): # -> str:
        ...



def assert_equal(actual, desired, err_msg=..., verbose=..., **kwds): # -> None:
    ...
