

import io
import os
import multiprocessing
import dpkt
from typing import List
from concurrent.futures import Future, ProcessPoolExecutor
from logging import debug, info

"""Loads a PCAP file and counts contents with various levels of storage"""
class PCAPParallel:

    def __init__(self, pcap_file: str, callback=..., split_size: int = ..., maximum_count: int = ..., pcap_filter: str | None = ..., maximum_cores: int | None = ...) -> List[io.BytesIO]:
        ...

    def set_split_size(self): # -> int | None:

        ...

    @staticmethod
    def open_maybe_compressed(filename): # -> GzipFile | TextIOWrapper[_WrappedBuffer] | BZ2File | LZMAFile | BufferedReader[_BufferedReaderStream]:

        ...

    def split(self) -> List[io.BytesIO] | List[Future]:

        ...

    def save_packets(self): # -> None:

        ...

    def dpkt_callback(self, timestamp: float, packet: bytes): # -> None:

        ...
