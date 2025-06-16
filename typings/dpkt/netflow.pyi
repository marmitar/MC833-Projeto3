

from . import dpkt

"""Cisco Netflow."""
class NetflowBase(dpkt.Packet):

    __hdr__ = ...
    def __len__(self):
        ...

    def __bytes__(self): # -> bytes:
        ...

    def unpack(self, buf): # -> None:
        ...

    class NetflowRecordBase(dpkt.Packet):

        def __len__(self):
            ...

        def __bytes__(self): # -> bytes:
            ...

        def unpack(self, buf): # -> None:
            ...





class Netflow1(NetflowBase):

    class NetflowRecord(NetflowBase.NetflowRecordBase):

        __hdr__ = ...




class Netflow5(NetflowBase):

    __hdr__ = ...
    class NetflowRecord(NetflowBase.NetflowRecordBase):

        __hdr__ = ...




class Netflow6(NetflowBase):

    __hdr__ = ...
    class NetflowRecord(NetflowBase.NetflowRecordBase):

        __hdr__ = ...




class Netflow7(NetflowBase):

    __hdr__ = ...
    class NetflowRecord(NetflowBase.NetflowRecordBase):

        __hdr__ = ...




def test_net_flow_v1_unpack(): # -> None:
    ...

def test_net_flow_v5_unpack(): # -> None:
    ...
