

from . import dpkt

"""Yahoo Messenger."""
class YHOO(dpkt.Packet):

    __hdr__ = ...
    __byte_order__ = ...


class YMSG(dpkt.Packet):
    __hdr__ = ...
