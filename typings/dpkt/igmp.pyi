

from . import dpkt

"""Internet Group Management Protocol."""
class IGMP(dpkt.Packet):

    __hdr__ = ...
    def __bytes__(self): # -> bytes:
        ...



def test_construction_no_sum(): # -> None:
    ...

def test_construction_sum_set(): # -> None:
    ...
