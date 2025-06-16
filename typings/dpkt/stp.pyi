

from . import dpkt

"""Spanning Tree Protocol."""
class STP(dpkt.Packet):

    __hdr__ = ...
    @property
    def age(self):
        ...

    @age.setter
    def age(self, age): # -> None:
        ...

    @property
    def max_age(self):
        ...

    @max_age.setter
    def max_age(self, max_age): # -> None:
        ...

    @property
    def hello(self):
        ...

    @hello.setter
    def hello(self, hello): # -> None:
        ...

    @property
    def fd(self):
        ...

    @fd.setter
    def fd(self, fd): # -> None:
        ...



def test_stp(): # -> None:
    ...

def test_properties(): # -> None:
    ...
