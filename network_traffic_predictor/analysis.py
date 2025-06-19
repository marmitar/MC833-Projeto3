"""
Perform Exploratory Data Analysis (EDA) on the preprocessed network data.
"""

import os
import sys
from argparse import ArgumentParser
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from signal import SIGINT
from typing import IO, Final

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

from network_traffic_predictor.utils import cli_colors
from network_traffic_predictor.utils.plotting import set_theme
from network_traffic_predictor.utils.schemas import PACKET_SCHEMA

set_theme()


def _extract_traffic(df: pl.DataFrame, *, log: IO[str]) -> pl.DataFrame:
    """
    Aggregate all traffic in each second and sum the trafficked bytes.
    """
    print('Aggregating data to create bytes_per_second time series...', file=log)
    return (
        df.lazy()
        .select('Timestamp', 'Size (bytes)')
        .group_by_dynamic('Timestamp', every='1s')
        .agg(bytes_per_second=pl.col('Size (bytes)').sum())
        .collect()
    )


def _descriptive_stats(traffic_per_sec: pl.DataFrame, basename: Path, *, log: IO[str]) -> None:
    """
    Generates a bytes_per_second time series and calculates its descriptive statistics.
    """
    print('--- Descriptive Statistics for Bytes per Second ---', file=log)
    stats = traffic_per_sec.select(kbps=pl.col('bytes_per_second') / 1024).describe()
    print(stats, file=log)
    stats.to_pandas().to_latex(
        buf=basename.parent / f'{basename.stem}.describe.tex',
        float_format='%.1f',
        index=False,
        escape=True,
    )


def _protocol_histogram(df: pl.DataFrame, basename: Path, *, log: IO[str]) -> None:
    """
    Creates and saves a histogram of packet sizes for TCP and UDP protocols.
    """
    _ = plt.figure(figsize=(12, 7))

    _ = sns.displot(
        df.select('Size (bytes)', Tipo='Type').filter(pl.col('Size (bytes)') <= 100).to_pandas(),
        x='Size (bytes)',
        hue='Tipo',
        kind='hist',
        discrete=True,
        binwidth=1,
        multiple='dodge',
    )
    plt.gca().set_yscale('log')

    _ = plt.title('Distribuição de Tamanhos de Pacote por Protocolo', fontsize=16)
    _ = plt.xlabel('Tamanho do Pacote (Bytes)', fontsize=12)
    _ = plt.ylabel('Quantidade de Pacotes (Escala Log)', fontsize=12)
    plt.tight_layout()

    output_path = basename.parent / f'{basename.stem}.protocol_dist.png'
    plt.savefig(output_path, dpi=300)
    print(f'Protocol distribution hstogram saved to {output_path}.', file=log)
    plt.close()


def _time_series(traffic_per_sec: pl.DataFrame, basename: Path, *, log: IO[str]) -> None:
    """
    Simple time series plot.
    """
    fig = plt.figure(figsize=(15, 6))
    _ = plt.plot(traffic_per_sec['Timestamp'], traffic_per_sec['bytes_per_second'] / 1024)
    _ = plt.title('Volume de Tráfego na Rede pelo Tempo', fontsize=16)
    _ = plt.xlabel('Tempo', fontsize=12)
    _ = plt.ylabel('Kibibytes por Segundo ', fontsize=12)

    fig.axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    # fig.axes[-1].xaxis.set_major_locator(mdates.SecondLocator(bysecond=(0, 30)))
    fig.autofmt_xdate()  # auto-rotate date labels

    output_path = basename.parent / f'{basename.stem}.time_series.png'
    plt.savefig(output_path, dpi=300)
    print(f'Time series saved to {output_path}', file=log)
    plt.close()


def _time_series_decomposition(traffic_per_sec: pl.DataFrame, basename: Path, *, log: IO[str]) -> None:
    """
    Decomposition of the time series using `statsmodel`.
    """
    PERIOD: Final = 60
    print(f'Time series decomposition: assuming period of {PERIOD} seconds.', file=log)
    if len(traffic_per_sec) < 2 * PERIOD:
        print(f'Warning: Time series is too short for seasonal decomposition with period={PERIOD}. Skipping.', file=log)
        return

    traffic_pd = traffic_per_sec.select('Timestamp', 'bytes_per_second').to_pandas().set_index('Timestamp')
    decomposition = seasonal_decompose(traffic_pd['bytes_per_second'], model='additive', period=PERIOD)
    fig = decomposition.plot()

    fig.set_size_inches(12, 9)
    _ = fig.suptitle('Decomposição da Série Temporal', y=1.01, fontsize=16)
    _ = fig.axes[0].set_title('Observado')
    _ = fig.axes[1].set_ylabel('Tendência')
    _ = fig.axes[2].set_ylabel('Sazonal')
    _ = fig.axes[3].set_ylabel('Residual')

    _ = fig.axes[-1].set_xlabel('Tempo')
    fig.axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    # fig.axes[-1].xaxis.set_major_locator(mdates.SecondLocator(bysecond=(0, 30)))
    fig.autofmt_xdate()  # auto-rotate date labels
    plt.tight_layout()

    output_path = basename.parent / f'{basename.stem}.decomposition.png'
    plt.savefig(output_path, dpi=300)
    print(f'Time series decomposition saved to {output_path}', file=log)
    plt.close()


def _autocorrelation(traffic_per_sec: pl.DataFrame, basename: Path, *, log: IO[str]) -> None:
    """
    Generate ACF and PACF plots to understand correlations.s
    """
    traffic_pd = traffic_per_sec.select('Timestamp', 'bytes_per_second').to_pandas().set_index('Timestamp')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))
    _ = plot_acf(traffic_pd['bytes_per_second'], ax=ax1, lags=40)
    ax1.set_title('Função de Autocorrelação (ACF)')
    _ = plot_pacf(traffic_pd['bytes_per_second'], ax=ax2, lags=40)
    ax2.set_title('Função de Autocorrelação Parcial (PACF)')
    plt.tight_layout()

    output_path = basename.parent / f'{basename.stem}.autocorrelation.png'
    plt.savefig(output_path, dpi=300)
    print(f'Autocorrelation saved to {output_path}', file=log)
    plt.close(fig)


def _run_all_analysis(*, input_file: Path, output_dir: Path, log: IO[str]) -> None:
    """
    Apply all proposed analysis on the data.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    basename = output_dir / input_file.name

    print(f'Analyzing {input_file.name}:', file=log)
    df = pl.read_parquet(input_file, schema=PACKET_SCHEMA)

    traffic_per_sec = _extract_traffic(df, log=log)
    _descriptive_stats(traffic_per_sec, basename, log=log)

    _protocol_histogram(df, basename, log=log)
    _time_series(traffic_per_sec, basename, log=log)
    _autocorrelation(traffic_per_sec, basename, log=log)
    _time_series_decomposition(traffic_per_sec, basename, log=log)


@contextmanager
def _open_log_file(*, quiet: bool) -> Iterator[IO[str]]:
    """
    Handle the log file for verbose and quiet output.
    """
    if quiet:
        with open(os.devnull, 'w') as null:
            yield null
    else:
        yield sys.stderr


def main() -> int:
    """
    Generate and visualize descriptive statistics from a processed parquet file.
    """
    parser = ArgumentParser(
        'analysis',
        description='Generate and visualize descriptive statistics from a processed parquet file.',
    )
    _ = parser.add_argument('parquet_file', type=Path, help='Generated by the preprocess script.')
    _ = parser.add_argument(
        '-o',
        '--output-dir',
        type=Path,
        default=Path('results'),
        help='Directory to save the output plots.',
    )
    _ = parser.add_argument('-q', '--quiet', action='store_true', help="Don't display progress.")
    _ = cli_colors.add_color_option(parser)

    args = parser.parse_intermixed_args()
    try:
        with _open_log_file(quiet=args.quiet) as log:
            _run_all_analysis(input_file=args.parquet_file, output_dir=args.output_dir, log=log)

        return 0
    except KeyboardInterrupt:
        return SIGINT


if __name__ == '__main__':
    sys.exit(main())
