

from . import dpkt

"""Dynamic Trunking Protocol."""
TRUNK_NAME = ...
MAC_ADDR = ...
class DTP(dpkt.Packet):

    __hdr__ = ...
    def unpack(self, buf): # -> None:
        ...

    def __bytes__(self): # -> bytes:
        ...



def test_creation(): # -> None:
    ...
