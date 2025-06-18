"""
Generate parquet file from raw PCAP data.
"""

import socket
import sys
from argparse import ArgumentParser
from io import BytesIO
from pathlib import Path
from signal import SIGINT
from typing import Literal
from warnings import warn

import dpkt
import polars as pl
from pcap_parallel import PCAPParallel

from network_traffic_predictor.utils import cli_colors
from network_traffic_predictor.utils.schemas import PACKET_WITHOUT_ID_SCHEMA


def _worker_process_chunk(file_handle: BytesIO) -> pl.DataFrame:
    """
    Transform a chunk of the PCAP file into structured data via Polars.
    """
    family: list[Literal['IPv4', 'IPv6']] = []
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
        except (dpkt.dpkt.UnpackError, AttributeError) as error:
            warn(f'{error}', stacklevel=1)
            continue

        if isinstance(eth.data, dpkt.ip.IP):
            address_family = socket.AF_INET
            family.append('IPv4')
        elif isinstance(eth.data, dpkt.ip6.IP6):
            address_family = socket.AF_INET6
            family.append('IPv6')
        else:
            continue

        ip = eth.data
        timestamps.append(int(timestamp * 1e9))
        source_ips.append(socket.inet_ntop(address_family, ip.src))
        dest_ips.append(socket.inet_ntop(address_family, ip.dst))
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
            'IP Family': family,
        },
        schema=PACKET_WITHOUT_ID_SCHEMA,
        strict=True,
    )


def _process_pcap_parallel(pcap_path: Path, *, quiet: bool) -> pl.DataFrame:
    """
    Parses a PCAP file in parallel using the pcap-parallel library.
    WARNING: This loads the entire PCAP file into memory.
    """
    if not quiet:
        print(f'Processing {pcap_path.name} with pcap-parallel...')

    ps = PCAPParallel(str(pcap_path), callback=_worker_process_chunk)
    df = pl.concat(future.result() for future in ps.split())
    return df.with_row_index('Packet', offset=1)


def main() -> int:
    """
    Generate parquet file from raw PCAP data.
    """
    parser = ArgumentParser('process', description='Generate parquet file from raw PCAP data.')
    _ = parser.add_argument('pcap_file', type=Path, help='Raw file to be processed into structured parquet.')
    _ = parser.add_argument('-q', '--quiet', action='store_true', help="Don't display progress.")
    _ = cli_colors.add_color_option(parser)

    args = parser.parse_intermixed_args()
    try:
        pcap_file: Path = args.pcap_file
        output_file = pcap_file.parent / f'{pcap_file.stem}.parquet'

        df = _process_pcap_parallel(pcap_file, quiet=args.quiet)
        df.write_parquet(output_file)
        print(df)

        return 0
    except KeyboardInterrupt:
        return SIGINT


if __name__ == '__main__':
    sys.exit(main())
