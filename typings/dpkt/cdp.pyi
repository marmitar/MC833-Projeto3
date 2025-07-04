

from . import dpkt

"""Cisco Discovery Protocol."""
CDP_DEVID = ...
CDP_ADDRESS = ...
CDP_PORTID = ...
CDP_CAPABILITIES = ...
CDP_VERSION = ...
CDP_PLATFORM = ...
CDP_IPPREFIX = ...
CDP_VTP_MGMT_DOMAIN = ...
CDP_NATIVE_VLAN = ...
CDP_DUPLEX = ...
CDP_TRUST_BITMAP = ...
CDP_UNTRUST_COS = ...
CDP_SYSTEM_NAME = ...
CDP_SYSTEM_OID = ...
CDP_MGMT_ADDRESS = ...
CDP_LOCATION = ...
class CDP(dpkt.Packet):

    __hdr__ = ...
    class TLV(dpkt.Packet):

        __hdr__ = ...
        def data_len(self): # -> int:
            ...

        def unpack(self, buf): # -> None:
            ...

        def __len__(self):
            ...

        def __bytes__(self): # -> bytes:
            ...



    class Address(TLV):
        __hdr__ = ...
        def data_len(self):
            ...



    class TLV_Addresses(TLV):
        __hdr__ = ...


    def unpack(self, buf): # -> None:
        ...

    def __len__(self):
        ...

    def __bytes__(self): # -> bytes:
        ...

    tlv_types = ...


def test_cdp(): # -> None:
    ...

def test_tlv(): # -> None:
    ...

def test_address(): # -> None:
    ...
