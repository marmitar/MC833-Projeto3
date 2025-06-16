

from . import dpkt

"""User Datagram Protocol."""
UDP_HDR_LEN = ...
UDP_PORT_MAX = ...
class UDP(dpkt.Packet):

    __hdr__ = ...

    @property
    def sport(self) -> int:
        ...

    @property
    def dport(self) -> int:
        ...
