

from . import dpkt

"""AOL Instant Messenger."""
class FLAP(dpkt.Packet):

    __hdr__ = ...
    def unpack(self, buf): # -> None:
        ...



class SNAC(dpkt.Packet):

    __hdr__ = ...


def tlv(buf): # -> tuple[Any, Any, Any, Any]:
    ...

def testAIM(): # -> None:
    ...

def testExceptions(): # -> None:
    ...
