

from . import dns, dpkt

"""Network Basic Input/Output System."""
def encode_name(name): # -> LiteralString:

    ...

def decode_name(nbname): # -> LiteralString:

    ...

NS_A = ...
NS_NS = ...
NS_NULL = ...
NS_NB = ...
NS_NBSTAT = ...
NS_IN = ...
NS_NAME_G = ...
NS_NAME_DRG = ...
NS_NAME_CNF = ...
NS_NAME_ACT = ...
NS_NAME_PRM = ...
nbstat_svcs = ...
def node_to_service_name(name_service_flags): # -> str:
    ...

class NS(dns.DNS):

    class Q(dns.DNS.Q):
        ...


    class RR(dns.DNS.RR):

        _node_name_struct = ...
        _node_name_len = ...
        def unpack_rdata(self, buf, off): # -> None:
            ...





class Session(dpkt.Packet):

    __hdr__ = ...


SSN_MESSAGE = ...
SSN_REQUEST = ...
SSN_POSITIVE = ...
SSN_NEGATIVE = ...
SSN_RETARGET = ...
SSN_KEEPALIVE = ...
class Datagram(dpkt.Packet):

    __hdr__ = ...


DGRAM_UNIQUE = ...
DGRAM_GROUP = ...
DGRAM_BROADCAST = ...
DGRAM_ERROR = ...
DGRAM_QUERY = ...
DGRAM_POSITIVE = ...
DGRAM_NEGATIVE = ...
def test_encode_name(): # -> None:
    ...

def test_decode_name(): # -> None:
    ...

def test_node_to_service_name(): # -> None:
    ...

def test_node_to_service_name_keyerror(): # -> None:
    ...

def test_rr(): # -> None:
    ...

def test_rr_nbstat(): # -> None:
    ...

def test_ns(): # -> None:
    ...
