

from . import dpkt

"""Routing Information Protocol."""
REQUEST = ...
RESPONSE = ...
class RIP(dpkt.Packet):

    __hdr__ = ...
    def unpack(self, buf): # -> None:
        ...

    def __len__(self):
        ...

    def __bytes__(self): # -> bytes:
        ...



class RTE(dpkt.Packet):
    __hdr__ = ...


class Auth(dpkt.Packet):
    __hdr__ = ...


def test_creation_with_auth(): # -> None:
    ...
