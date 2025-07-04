

from . import dpkt

"""Hypertext Transfer Protocol Version 2."""
HTTP2_PREFACE = ...
HTTP2_FRAME_DATA = ...
HTTP2_FRAME_HEADERS = ...
HTTP2_FRAME_PRIORITY = ...
HTTP2_FRAME_RST_STREAM = ...
HTTP2_FRAME_SETTINGS = ...
HTTP2_FRAME_PUSH_PROMISE = ...
HTTP2_FRAME_PING = ...
HTTP2_FRAME_GOAWAY = ...
HTTP2_FRAME_WINDOW_UPDATE = ...
HTTP2_FRAME_CONTINUATION = ...
HTTP2_FLAG_END_STREAM = ...
HTTP2_FLAG_ACK = ...
HTTP2_FLAG_END_HEADERS = ...
HTTP2_FLAG_PADDED = ...
HTTP2_FLAG_PRIORITY = ...
HTTP2_SETTINGS_HEADER_TABLE_SIZE = ...
HTTP2_SETTINGS_ENABLE_PUSH = ...
HTTP2_SETTINGS_MAX_CONCURRENT_STREAMS = ...
HTTP2_SETTINGS_INITIAL_WINDOW_SIZE = ...
HTTP2_SETTINGS_MAX_FRAME_SIZE = ...
HTTP2_SETTINGS_MAX_HEADER_LIST_SIZE = ...
HTTP2_NO_ERROR = ...
HTTP2_PROTOCOL_ERROR = ...
HTTP2_INTERNAL_ERROR = ...
HTTP2_FLOW_CONTROL_ERROR = ...
HTTP2_SETTINGS_TIMEOUT = ...
HTTP2_STREAM_CLOSED = ...
HTTP2_FRAME_SIZE_ERROR = ...
HTTP2_REFUSED_STREAM = ...
HTTP2_CANCEL = ...
HTTP2_COMPRESSION_ERROR = ...
HTTP2_CONNECT_ERROR = ...
HTTP2_ENHANCE_YOUR_CALM = ...
HTTP2_INADEQUATE_SECURITY = ...
HTTP2_HTTP_1_1_REQUIRED = ...
error_code_str = ...
class HTTP2Exception(Exception):
    ...


class Preface(dpkt.Packet):
    __hdr__ = ...
    def unpack(self, buf): # -> None:
        ...



class Frame(dpkt.Packet):

    __hdr__ = ...
    def unpack(self, buf): # -> None:
        ...

    @property
    def length(self): # -> Any:
        ...



class Priority(dpkt.Packet):

    __hdr__ = ...
    def unpack(self, buf): # -> None:
        ...



class Setting(dpkt.Packet):

    __hdr__ = ...


class PaddedFrame(Frame):

    def unpack(self, buf): # -> None:
        ...



class DataFrame(PaddedFrame):

    @property
    def payload(self): # -> bytes:
        ...



class HeadersFrame(PaddedFrame):

    def unpack(self, buf): # -> None:
        ...



class PriorityFrame(Frame):

    def unpack(self, buf): # -> None:
        ...



class RSTStreamFrame(Frame):

    def unpack(self, buf): # -> None:
        ...



class SettingsFrame(Frame):

    def unpack(self, buf): # -> None:
        ...



class PushPromiseFrame(PaddedFrame):

    def unpack(self, buf): # -> None:
        ...



class PingFrame(Frame):

    def unpack(self, buf): # -> None:
        ...



class GoAwayFrame(Frame):

    def unpack(self, buf): # -> None:
        ...



class WindowUpdateFrame(Frame):

    def unpack(self, buf): # -> None:
        ...



class ContinuationFrame(Frame):

    def unpack(self, buf): # -> None:
        ...



FRAME_TYPES = ...
class FrameFactory:
    def __new__(cls, buf): # -> ContinuationFrame | DataFrame | GoAwayFrame | HeadersFrame | PingFrame | PriorityFrame | PushPromiseFrame | RSTStreamFrame | SettingsFrame | WindowUpdateFrame:
        ...



def frame_multi_factory(buf, preface=...): # -> tuple[list[Any], Literal[0]] | tuple[list[Any], int]:

    ...

class TestFrame:

    @classmethod
    def setup_class(cls): # -> None:
        ...

    def test_frame(self): # -> None:
        ...

    def test_data(self): # -> None:
        ...

    def test_headers(self): # -> None:
        ...

    def test_priority(self): # -> None:
        ...

    def test_rst_stream(self): # -> None:
        ...

    def test_settings(self): # -> None:
        ...

    def test_push_promise(self): # -> None:
        ...

    def test_ping(self): # -> None:
        ...

    def test_goaway(self): # -> None:
        ...

    def test_window_update(self): # -> None:
        ...

    def test_continuation(self): # -> None:
        ...

    def test_factory(self): # -> None:
        ...

    def test_preface(self): # -> None:
        ...

    def test_multi(self): # -> None:
        ...
