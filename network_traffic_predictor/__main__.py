import socket
from io import BytesIO
from pathlib import Path
from typing import Literal

import dpkt
import polars as pl
from pcap_parallel import PCAPParallel


def worker_process_chunk(file_handle: BytesIO) -> pl.DataFrame:
    """
    This function is executed by each worker process on a chunk of the PCAP file.
    It returns a dictionary of lists, which is efficient to merge later.
    """
    # Each worker gets its own lists to avoid any shared state issues.
    timestamps: list[int] = []
    source_ips: list[str] = []
    dest_ips: list[str] = []
    protocols: list[int] = []
    sizes: list[int] = []
    source_ports: list[int | None] = []
    dest_ports: list[int | None] = []
    types: list[Literal['TCP', 'UDP', 'other']] = []

    for timestamp, buf in dpkt.pcap.Reader(file_handle):
        try:
            eth = dpkt.ethernet.Ethernet(buf)
            if isinstance(eth.data, dpkt.ip.IP):
                ip = eth.data
                timestamps.append(timestamp)
                source_ips.append(socket.inet_ntoa(ip.src))
                dest_ips.append(socket.inet_ntoa(ip.dst))
                protocols.append(ip.p)
                sizes.append(len(buf))

                if isinstance(ip.data, dpkt.tcp.TCP):
                    types.append('TCP')
                    source_ports.append(ip.data.sport)
                    dest_ports.append(ip.data.dport)
                elif isinstance(ip.data, dpkt.udp.UDP):
                    types.append('UDP')
                    source_ports.append(ip.data.sport)
                    dest_ports.append(ip.data.dport)
                else:
                    types.append('other')
                    source_ports.append(None)
                    dest_ports.append(None)
        except (dpkt.dpkt.UnpackError, AttributeError):
            continue

    # THE FIX: Create a DataFrame inside the worker with a defined schema.
    # This guarantees that every returned chunk has the same data types.
    return pl.DataFrame(
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
        schema={
            'Timestamp': pl.Float64,
            'Source IP': pl.String,
            'Destination IP': pl.String,
            'Protocol': pl.UInt8,
            'Size (bytes)': pl.UInt32,
            # Ports can be null, so we use integer types that support nulls.
            'Source Port': pl.UInt16,
            'Destination Port': pl.UInt16,
            'Type': pl.String,
        },
    )


def process_pcap_parallel(pcap_path: Path, parquet_output_path: Path):
    """
    Parses a PCAP file in parallel using the pcap-parallel library.
    WARNING: This loads the entire PCAP file into memory.
    """
    print(f'Processing {pcap_path.name} with pcap-parallel...')

    # Initialize the parallel processor.
    # We can specify max_cores to match your Ryzen 7 7700X.
    ps = PCAPParallel(
        str(pcap_path),
        callback=worker_process_chunk,
    )

    # This call blocks until all chunks are processed by the worker pool.
    # It returns a list of Future objects.
    future_results = ps.split()
    print(f'PCAP file split and processed by {len(future_results)} workers.')

    # Collect results from all workers.
    # .result() blocks until that specific future is done.
    list_of_dicts = [future.result() for future in future_results]

    print('Merging results and creating Polars DataFrame...')

    # Polars can efficiently create a DataFrame from a list of dictionaries
    df = pl.concat(list_of_dicts)

    # Add packet numbers and convert timestamp
    df = df.with_row_index('Packet', offset=1).with_columns(pl.from_epoch('Timestamp', time_unit='s'))

    print(f'Writing output to {parquet_output_path}...')
    df.write_parquet(parquet_output_path)
    print('Done.')

    return df


# --- Main Execution ---
if __name__ == '__main__':
    # Make sure to decompress the .dump.gz file to a .pcap or .dump file first
    pcap_file = Path('data/200701251400.dump')

    if not pcap_file.exists():
        print(f"Error: Input file '{pcap_file}' not found.")
        print('Please decompress the .dump.gz file first.')
    else:
        # --- CHOOSE WHICH VERSION TO RUN ---

        # Option 2: Run the parallel version (requires more RAM)
        parquet_out_parallel = Path(f'{pcap_file.stem}_parallel.parquet')
        df_parallel = process_pcap_parallel(pcap_file, parquet_out_parallel)
        print('\n--- Parallel DataFrame Head ---')
        print(df_parallel.head())
