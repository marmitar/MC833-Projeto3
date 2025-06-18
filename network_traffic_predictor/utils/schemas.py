"""
Shared polars and parquet schemas.
"""

from datetime import UTC
from typing import Final

import polars as pl

# Partial schema during PCAP processing.
PACKET_WITHOUT_ID_SCHEMA: Final = pl.Schema({
    'Timestamp': pl.Datetime(time_unit='ns', time_zone=UTC),
    'Source IP': pl.String,
    'Destination IP': pl.String,
    'Protocol': pl.UInt8,
    'Size (bytes)': pl.UInt32,
    'Source Port': pl.UInt16,
    'Destination Port': pl.UInt16,
    'Type': pl.Enum(['TCP', 'UDP', 'Outro']),
})

# Schema used for the parquet files.
PACKET_SCHEMA: Final = pl.Schema({
    'Packet': pl.UInt32,
    **PACKET_WITHOUT_ID_SCHEMA,
})
