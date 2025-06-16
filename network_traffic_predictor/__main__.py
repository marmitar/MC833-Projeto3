import socket
import sys
from datetime import UTC
from io import BytesIO
from pathlib import Path
from typing import Final, Literal

import dpkt
import polars as pl
from pcap_parallel import PCAPParallel

_PKT_SCHEMA: Final = pl.Schema({
    'Timestamp': pl.Datetime(time_unit='ns', time_zone=UTC),
    'Source IP': pl.String,
    'Destination IP': pl.String,
    'Protocol': pl.UInt8,
    'Size (bytes)': pl.UInt32,
    'Source Port': pl.UInt16,
    'Destination Port': pl.UInt16,
    'Type': pl.Enum(['TCP', 'UDP', 'Outro']),
})


def _seconds_to_nanos(timestamp: float) -> int:
    return int(timestamp * 1e9)


def _worker_process_chunk(file_handle: BytesIO) -> pl.DataFrame:
    timestamps: list[int] = []
    source_ips: list[str] = []
    dest_ips: list[str] = []
    protocols: list[int] = []
    sizes: list[int] = []
    source_ports: list[int | None] = []
    dest_ports: list[int | None] = []
    types: list[Literal['TCP', 'UDP', 'Outro']] = []

    for timestamp, buf in dpkt.pcap.Reader(file_handle):
        try:
            eth = dpkt.ethernet.Ethernet(buf)
            if isinstance(eth.data, dpkt.ip.IP):
                ip = eth.data
                timestamps.append(_seconds_to_nanos(timestamp))
                source_ips.append(socket.inet_ntoa(ip.src))
                dest_ips.append(socket.inet_ntoa(ip.dst))
                protocols.append(ip.p)
                sizes.append(len(buf))

                match ip.data:
                    case dpkt.tcp.TCP() as tcp:
                        types.append('TCP')
                        source_ports.append(tcp.sport)
                        dest_ports.append(tcp.dport)
                    case dpkt.udp.UDP() as udp:
                        types.append('UDP')
                        source_ports.append(udp.sport)
                        dest_ports.append(udp.dport)
                    case _:
                        types.append('Outro')
                        source_ports.append(None)
                        dest_ports.append(None)

        except (dpkt.dpkt.UnpackError, AttributeError):
            continue

    return pl.from_dict(
        {
            'Timestamp': timestamps,
            'Source IP': source_ips,
            'Destination IP': dest_ips,
            'Protocol': protocols,
            'Size (bytes)': sizes,
            'Source Port': source_ports,
            'Destination Port': dest_ports,
            'Type': types,
        },
        schema=_PKT_SCHEMA,
        strict=True,
    )


def process_pcap_parallel(pcap_path: Path):
    """
    Parses a PCAP file in parallel using the pcap-parallel library.
    WARNING: This loads the entire PCAP file into memory.
    """
    print(f'Processing {pcap_path.name} with pcap-parallel...')

    ps = PCAPParallel(
        str(pcap_path),
        callback=_worker_process_chunk,
    )

    return (
        pl.concat(future.result() for future in ps.split())
        .lazy()
        .with_row_index('Packet', offset=1)
        .with_columns(pl.from_epoch('Timestamp', time_unit='s'))
        .collect()
    )


if __name__ == '__main__':
    pcap_file = Path('data/200701251400.dump')

    if not pcap_file.exists():
        print(f"Error: Input file '{pcap_file}' not found.")
        print('Please decompress the .dump.gz file first.')
        sys.exit(-1)

    df = process_pcap_parallel(pcap_file)
    df.write_parquet(Path(f'{pcap_file.stem}_parallel.parquet'))
    print(df)
