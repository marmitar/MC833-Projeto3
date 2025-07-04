"""
This type stub file was generated by pyright.
"""

"""Vendoered from
https://github.com/pypa/packaging/blob/main/packaging/_structures.py
"""
class InfinityType:
    def __repr__(self) -> str:
        ...

    def __hash__(self) -> int:
        ...

    def __lt__(self, other: object) -> bool:
        ...

    def __le__(self, other: object) -> bool:
        ...

    def __eq__(self, other: object) -> bool:
        ...

    def __ne__(self, other: object) -> bool:
        ...

    def __gt__(self, other: object) -> bool:
        ...

    def __ge__(self, other: object) -> bool:
        ...

    def __neg__(self: object) -> NegativeInfinityType:
        ...



Infinity = ...
class NegativeInfinityType:
    def __repr__(self) -> str:
        ...

    def __hash__(self) -> int:
        ...

    def __lt__(self, other: object) -> bool:
        ...

    def __le__(self, other: object) -> bool:
        ...

    def __eq__(self, other: object) -> bool:
        ...

    def __ne__(self, other: object) -> bool:
        ...

    def __gt__(self, other: object) -> bool:
        ...

    def __ge__(self, other: object) -> bool:
        ...

    def __neg__(self: object) -> InfinityType:
        ...



NegativeInfinity = ...
