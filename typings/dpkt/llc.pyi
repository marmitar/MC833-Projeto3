

from . import dpkt

class LLC(dpkt.Packet):

    __hdr__ = ...
    @property
    def is_snap(self):
        ...

    def unpack(self, buf): # -> None:
        ...

    def pack_hdr(self): # -> bytes:
        ...

    def __len__(self):
        ...



def test_llc(): # -> None:
    ...

def test_unpack_sap_ip(): # -> None:
    ...

def test_unpack_exception_handling(): # -> None:
    ...

def test_pack_hdr_invalid_class(): # -> None:
    class InvalidClass(dpkt.Packet):
        ...
