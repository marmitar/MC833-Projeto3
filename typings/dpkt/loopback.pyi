

from . import dpkt

"""Platform-dependent loopback header."""
class Loopback(dpkt.Packet):

    __hdr__ = ...
    __byte_order__ = ...
    def unpack(self, buf): # -> None:
        ...



def test_ethernet_unpack(): # -> None:
    ...

def test_ip_unpack(): # -> None:
    ...

def test_ip6_unpack(): # -> None:
    ...
