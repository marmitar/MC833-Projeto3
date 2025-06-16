import subprocess
from pathlib import Path

import polars as pl


def process_pcap_with_tshark(pcap_path: Path, parquet_output_path: Path):
    """
    Uses tshark to process a PCAP file and saves the result as Parquet.
    """
    print(f'Processing {pcap_path.name} with tshark...')

    # Define the exact fields you want tshark to extract.
    # See `tshark -G fields` for all possible fields.
    fields = [
        'frame.time_epoch',
        'ip.src',
        'ip.dst',
        'ip.proto',
        'frame.len',  # 'len(pkt)' in scapy includes the whole frame
        'tcp.srcport',
        'tcp.dstport',
        'udp.srcport',
        'udp.dstport',
    ]

    # Build the tshark command
    tshark_cmd = [
        'tshark',
        '-r',
        str(pcap_path),  # Input file
        '-T',
        'fields',  # Output in fields format
        '-E',
        'header=y',  # Add a CSV header
        '-E',
        'separator=,',  # Use comma as separator
        '-E',
        'quote=d',  # Use double quotes
        '-E',
        'occurrence=f',  # Only show the first occurrence of a field
    ]
    # Add each field to the command
    for field in fields:
        tshark_cmd.extend(['-e', field])

    # Run the command and capture the output
    proc = subprocess.run(tshark_cmd, capture_output=True, text=True, check=True)
    csv_data = proc.stdout

    # Use Polars to read the CSV data from the string output.
    # Polars is extremely fast at this.
    print('tshark finished. Reading CSV data with Polars...')
    df = pl.read_csv(
        source=csv_data.encode('utf-8'),  # read_csv wants bytes
        has_header=True,
    )

    # --- Data Cleaning and Transformation in Polars ---
    # Rename columns to match the original DataFrame
    df = df.rename({
        'frame.time_epoch': 'Timestamp',
        'ip.src': 'Source IP',
        'ip.dst': 'Destination IP',
        'ip.proto': 'Protocol',
        'frame.len': 'Size (bytes)',
    })

    # Convert timestamp from epoch to datetime
    df = df.with_columns(pl.from_epoch('Timestamp', time_unit='s'))

    # Combine TCP and UDP ports into single columns
    df = df.with_columns(
        pl.coalesce(['tcp.srcport', 'udp.srcport']).alias('Source Port'),
        pl.coalesce(['tcp.dstport', 'udp.dstport']).alias('Destination Port'),
    ).drop(['tcp.srcport', 'udp.srcport', 'tcp.dstport', 'udp.dstport'])

    print(f'Writing output to {parquet_output_path}...')
    df.write_parquet(parquet_output_path)
    print('Done.')
    return df


# --- Usage ---
# Assuming '200701251400.pcap' is the decompressed file from your notebook
pcap_file = Path('data/200701251400.dump')
parquet_file = Path('200701251400_fast.parquet')
if pcap_file.exists():
    df_fast = process_pcap_with_tshark(pcap_file, parquet_file)
    print(df_fast.head())
else:
    print(f'{pcap_file} not found. Please decompress it first.')
