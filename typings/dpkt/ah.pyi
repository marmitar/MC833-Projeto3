

from . import dpkt

"""Authentication Header."""
class AH(dpkt.Packet):

    __hdr__ = ...
    auth = ...
    def unpack(self, buf): # -> None:
        ...

    def __len__(self):
        ...

    def __bytes__(self): # -> bytes:
        ...



def test_default_creation(): # -> None:
    ...

def test_creation_from_buf(): # -> None:
    ...
