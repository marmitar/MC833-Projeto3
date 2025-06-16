

from . import dpkt

"""Linux libpcap "cooked v2" capture encapsulation."""
class SLL2(dpkt.Packet):

    __hdr__ = ...
    _typesw = ...
    def unpack(self, buf): # -> None:
        ...



def test_sll2(): # -> None:
    ...
