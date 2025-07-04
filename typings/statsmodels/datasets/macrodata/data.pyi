"""
This type stub file was generated by pyright.
"""

"""United States Macroeconomic data"""
__docformat__ = ...
COPYRIGHT = ...
TITLE = ...
SOURCE = ...
DESCRSHORT = ...
DESCRLONG = ...
NOTE = ...
def load_pandas(): # -> Dataset:
    ...

def load(): # -> Dataset:
    """
    Load the US macro data and return a Dataset class.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.

    Notes
    -----
    The macrodata Dataset instance does not contain endog and exog attributes.
    """
    ...

variable_names = ...
def __str__() -> str:
    ...
