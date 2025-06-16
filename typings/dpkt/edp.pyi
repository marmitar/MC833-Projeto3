

import dpkt

"""Extreme Discovery Protocol."""
class EDP(dpkt.Packet):
    __hdr__ = ...
    def __bytes__(self): # -> bytes:
        ...



class TestEDP:

    @classmethod
    def setup_class(cls): # -> None:
        ...

    def test_version(self): # -> None:
        ...

    def test_reserved(self): # -> None:
        ...

    def test_hlen(self): # -> None:
        ...

    def test_sum(self): # -> None:
        ...

    def test_seq(self): # -> None:
        ...

    def test_mid(self): # -> None:
        ...

    def test_mac(self): # -> None:
        ...

    def test_bytes(self): # -> None:
        ...
