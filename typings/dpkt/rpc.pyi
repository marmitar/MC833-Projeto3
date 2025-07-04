

from . import dpkt

"""Remote Procedure Call."""
CALL = ...
REPLY = ...
AUTH_NULL = ...
AUTH_UNIX = ...
AUTH_SHORT = ...
AUTH_DES = ...
MSG_ACCEPTED = ...
MSG_DENIED = ...
SUCCESS = ...
PROG_UNAVAIL = ...
PROG_MISMATCH = ...
PROC_UNAVAIL = ...
GARBAGE_ARGS = ...
SYSTEM_ERR = ...
RPC_MISMATCH = ...
AUTH_ERROR = ...
class RPC(dpkt.Packet):

    __hdr__ = ...
    class Auth(dpkt.Packet):
        __hdr__ = ...
        def unpack(self, buf): # -> None:
            ...

        def __len__(self): # -> int:
            ...

        def __bytes__(self): # -> bytes:
            ...



    class Call(dpkt.Packet):
        __hdr__ = ...
        def unpack(self, buf): # -> None:
            ...

        def __len__(self): # -> int:
            ...

        def __bytes__(self): # -> bytes:
            ...



    class Reply(dpkt.Packet):
        __hdr__ = ...
        class Accept(dpkt.Packet):
            __hdr__ = ...
            def unpack(self, buf): # -> None:
                ...

            def __len__(self): # -> int:
                ...

            def __bytes__(self): # -> bytes:
                ...



        class Reject(dpkt.Packet):
            __hdr__ = ...
            def unpack(self, buf): # -> None:
                ...

            def __len__(self): # -> int:
                ...

            def __bytes__(self): # -> bytes:
                ...



        def unpack(self, buf): # -> None:
            ...



    def unpack(self, buf): # -> None:
        ...



def unpack_xdrlist(cls, buf): # -> list[Any]:
    ...

def pack_xdrlist(*args): # -> bytes:
    ...

def test_auth(): # -> None:
    ...

def test_call(): # -> None:
    ...

def test_reply(): # -> None:
    ...

def test_accept(): # -> None:
    ...

def test_reject(): # -> None:
    ...

def test_rpc(): # -> None:
    ...
