

from . import dpkt

"""Internet Protocol, version 6."""
EXT_HDRS = ...
class IP6(dpkt.Packet):

    __hdr__ = ...
    __bit_fields__ = ...
    __pprint_funcs__ = ...
    _protosw = ...
    def unpack(self, buf): # -> None:
        ...

    def headers_str(self): # -> tuple[Any, bytes]:
        ...

    def __bytes__(self): # -> bytes:
        ...

    def __len__(self):
        ...

    @classmethod
    def set_proto(cls, p, pktclass): # -> None:
        ...

    @classmethod
    def get_proto(cls, p):
        ...

    @property
    def src(self) -> bytes:
        ...

    @property
    def dst(self) -> bytes:
        ...

    @property
    def p(self) -> int:
        ...



class IP6ExtensionHeader(dpkt.Packet):

    ...


class IP6OptsHeader(IP6ExtensionHeader):
    __hdr__ = ...
    def unpack(self, buf): # -> None:
        ...



class IP6HopOptsHeader(IP6OptsHeader):
    ...


class IP6DstOptsHeader(IP6OptsHeader):
    ...


class IP6RoutingHeader(IP6ExtensionHeader):
    __hdr__ = ...
    __bit_fields__ = ...
    def unpack(self, buf): # -> None:
        ...



class IP6FragmentHeader(IP6ExtensionHeader):
    __hdr__ = ...
    __bit_fields__ = ...
    def unpack(self, buf): # -> None:
        ...



class IP6AHHeader(IP6ExtensionHeader):
    __hdr__ = ...
    def unpack(self, buf): # -> None:
        ...



class IP6ESPHeader(IP6ExtensionHeader):
    __hdr__ = ...
    def unpack(self, buf): # -> None:
        ...



EXT_HDRS_CLS = ...
def test_ipg(): # -> None:
    ...

def test_dict(): # -> None:
    ...

def test_ip6_routing_header(): # -> None:
    ...

def test_ip6_fragment_header(): # -> None:
    ...

def test_ip6_options_header(): # -> None:
    ...

def test_ip6_ah_header(): # -> None:
    ...

def test_ip6_esp_header(): # -> None:
    ...

def test_ip6_extension_headers(): # -> None:
    ...

def test_ip6_all_extension_headers(): # -> None:
    ...

def test_ip6_gen_tcp_ack(): # -> None:
    ...

def test_ip6_opts(): # -> None:
    ...

def test_ip6_routing_properties(): # -> None:
    ...

def test_ip6_fragment_properties(): # -> None:
    ...

def test_ip6_properties(): # -> None:
    ...

def test_proto_accessors(): # -> None:
    class Proto:
        ...



def test_ip6_fragment_no_decode(): # -> None:
    ...
