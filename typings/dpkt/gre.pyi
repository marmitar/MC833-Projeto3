

from . import dpkt

"""Generic Routing Encapsulation."""
GRE_CP = ...
GRE_RP = ...
GRE_KP = ...
GRE_SP = ...
GRE_SS = ...
GRE_AP = ...
GRE_opt_fields = ...
class GRE(dpkt.Packet):

    __hdr__ = ...
    sre = ...
    @property
    def v(self):
        ...

    @v.setter
    def v(self, v): # -> None:
        ...

    @property
    def recur(self):

        ...

    @recur.setter
    def recur(self, v): # -> None:
        ...

    class SRE(dpkt.Packet):
        __hdr__ = ...
        def unpack(self, buf): # -> None:
            ...



    def opt_fields_fmts(self): # -> tuple[list[str], list[str]]:
        ...

    def unpack(self, buf): # -> None:
        ...

    def __len__(self):
        ...

    def __bytes__(self): # -> bytes:
        ...



def test_gre_v1(): # -> None:
    ...

def test_gre_len(): # -> None:
    ...

def test_gre_accessors(): # -> None:
    ...

def test_sre_creation(): # -> None:
    ...

def test_gre_nested_sre(): # -> None:
    ...

def test_gre_next_layer(): # -> None:
    ...
