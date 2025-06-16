import socket
from pathlib import Path

import dpkt
import polars as pl


def process_pcap_with_dpkt(pcap_path: Path, parquet_output_path: Path):
    """
    Parses a PCAP file using the lightweight dpkt library and aggregates
    the results with the high-performance Polars DataFrame library.

    Args:
        pcap_path: The path to the input PCAP file.
        parquet_output_path: The path to save the final Parquet file.

    Returns:
        A Polars DataFrame containing the processed packet data.
    """
    print(f'Processing {pcap_path.name} with dpkt and polars...')

    # 1. Use simple lists for fast data accumulation.
    # This is much more efficient than appending dictionaries.
    timestamps = []
    source_ips = []
    dest_ips = []
    protocols = []
    sizes = []
    source_ports = []
    dest_ports = []
    types = []

    packet_count = 0
    # Limit processing for demonstration, set to a very high number to process all
    # N = 9999999
    N = float('inf')

    try:
        with open(pcap_path, 'rb') as f:
            # dpkt's PcapReader is an efficient iterator over the file
            pcap = dpkt.pcap.Reader(f)

            for timestamp, buf in pcap:
                packet_count += 1
                if packet_count > N:
                    break

                # Unpack the Ethernet frame (level 2)
                # We need to know the link type. Common ones are DLT_EN10MB (Ethernet)
                # and DLT_RAW (raw IP). Let's assume Ethernet for MAWI data.
                try:
                    eth = dpkt.ethernet.Ethernet(buf)
                except dpkt.dpkt.UnpackError:
                    # Skip packets that can't be unpacked as Ethernet
                    continue

                # Ensure the packet contains IP data (level 3)
                if not isinstance(eth.data, dpkt.ip.IP):
                    continue

                ip = eth.data

                # --- Append data to our lists ---
                timestamps.append(timestamp)
                # Convert packed binary IP addresses to human-readable strings
                source_ips.append(socket.inet_ntoa(ip.src))
                dest_ips.append(socket.inet_ntoa(ip.dst))
                protocols.append(ip.p)
                sizes.append(len(buf))  # len(buf) is equivalent to len(pkt)

                # Check for TCP or UDP payload (level 4)
                if isinstance(ip.data, dpkt.tcp.TCP):
                    tcp = ip.data
                    source_ports.append(tcp.sport)
                    dest_ports.append(tcp.dport)
                    types.append('TCP')
                elif isinstance(ip.data, dpkt.udp.UDP):
                    udp = ip.data
                    source_ports.append(udp.sport)
                    dest_ports.append(udp.dport)
                    types.append('UDP')
                else:
                    source_ports.append(None)
                    dest_ports.append(None)
                    types.append('Other')
    except Exception as e:
        print(f'An error occurred while reading the PCAP file: {e}')
        return None

    print(f'PCAP parsing complete. Processed {packet_count} packets.')
    print('Creating Polars DataFrame...')

    # 2. Create a Polars DataFrame directly from the lists.
    # This is the most efficient way to construct a DataFrame.
    df = pl.DataFrame({
        'Packet': range(1, len(timestamps) + 1),
        'Timestamp': timestamps,
        'Source IP': source_ips,
        'Destination IP': dest_ips,
        'Protocol': protocols,
        'Size (bytes)': sizes,
        'Source Port': source_ports,
        'Destination Port': dest_ports,
        'Type': types,
    })

    # Convert timestamp from epoch to datetime
    df = df.with_columns(pl.from_epoch('Timestamp', time_unit='s'))

    print(f'Writing output to {parquet_output_path}...')
    df.write_parquet(parquet_output_path)
    print('Done.')

    return df


# --- Example Usage ---
# Assumes '200701251400.pcap' is the decompressed file from your notebook
pcap_file = Path('data/200701251400.dump')
parquet_file = Path('200701251400_dpkt.parquet')

if pcap_file.exists():
    df_dpkt = process_pcap_with_dpkt(pcap_file, parquet_file)
    if df_dpkt is not None:
        print('\n--- Final DataFrame Head ---')
        print(df_dpkt.head())
else:
    print(f'Error: {pcap_file} not found.')
