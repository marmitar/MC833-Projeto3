


class Error(Exception):
    ...


class UnpackError(Error):
    ...


class NeedData(UnpackError):
    ...


class PackError(Error):
    ...


class _MetaPacket(type):
    def __new__(cls, clsname, clsbases, clsdict): # -> Self:
        ...



class Packet(_MetaPacket("Temp", (object, ), {})):

    def __init__(self, *args, **kwargs) -> None:

        ...

    def __len__(self):
        ...

    def __iter__(self): # -> Generator[tuple[Any, Any], None, None]:
        ...

    def __getitem__(self, kls): # -> Packet:

        ...

    def __contains__(self, kls): # -> bool:

        ...

    def __repr__(self): # -> str:
        ...

    def pprint(self, indent=...): # -> None:

        ...

    def __str__(self) -> str:
        ...

    def __bytes__(self): # -> bytes:
        ...

    def pack_hdr(self): # -> bytes:

        ...

    def pack(self): # -> bytes:

        ...

    def unpack(self, buf): # -> None:

        ...



__vis_filter = ...
def hexdump(buf, length=...): # -> LiteralString:

    ...

def in_cksum_add(s, buf):
    ...

def in_cksum_done(s): # -> Any:
    ...

def in_cksum(buf): # -> Any:

    ...

def test_utils(): # -> None:
    ...

def test_getitem_contains(): # -> None:
    class Foo(Packet):
        ...


    class Bar(Packet):
        ...


    class Baz(Packet):
        ...


    class Zeb(Packet):
        ...



def test_pack_hdr_overflow(): # -> None:

    class Foo(Packet):
        ...



def test_bit_fields_overflow(): # -> None:

    class Foo(Packet):
        ...



def test_pack_hdr_tuple(): # -> None:

    class Foo(Packet):
        ...



def test_unpacking_failure(): # -> None:
    class TestPacket(Packet):
        ...



def test_repr(): # -> None:

    class TestPacket(Packet):
        ...
