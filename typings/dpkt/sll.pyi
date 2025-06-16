

from . import dpkt

"""Linux libpcap "cooked" capture encapsulation."""
class SLL(dpkt.Packet):

    __hdr__ = ...
    _typesw = ...
    def unpack(self, buf): # -> None:
        ...



def test_sll(): # -> None:
    ...
