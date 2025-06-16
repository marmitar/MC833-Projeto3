

from . import dpkt

"""ISO Transport Service on top of the TCP (TPKT)."""
class TPKT(dpkt.Packet):

    __hdr__ = ...
