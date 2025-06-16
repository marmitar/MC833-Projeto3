

from . import dpkt

"""Transparent Network Substrate."""
class TNS(dpkt.Packet):

    __hdr__ = ...
    def unpack(self, buf): # -> None:
        ...



def test_tns(): # -> None:
    ...
