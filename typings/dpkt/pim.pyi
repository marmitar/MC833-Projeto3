

from . import dpkt

"""Protocol Independent Multicast."""
class PIM(dpkt.Packet):

    __hdr__ = ...
    __bit_fields__ = ...
    def __bytes__(self): # -> bytes:
        ...



def test_pim(): # -> None:
    ...
