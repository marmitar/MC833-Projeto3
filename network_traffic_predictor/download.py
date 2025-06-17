"""Download files listed in data sources."""

import asyncio
import glob
import gzip
import os
import re
import struct
import sys
from argparse import ArgumentParser, BooleanOptionalAction
from collections.abc import AsyncIterator, Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, asynccontextmanager
from io import IOBase
from itertools import cycle
from pathlib import Path
from signal import SIGINT
from traceback import print_exception
from typing import Final, Literal
from urllib.parse import ParseResult, urlparse
from urllib.request import urlopen

import colored_traceback
from matplotlib import colormaps
from matplotlib.colors import to_hex
from tqdm import tqdm

# --- Colored Output ---


def _colormap_cycle(name: str) -> Iterator[str]:
    """Get the colormap as a cycle."""
    cmap = colormaps.get_cmap(name)
    colors = (to_hex(cmap(i)) for i in range(cmap.N))
    return cycle(colors)


_TAB10_COLORS = _colormap_cycle('tab10')


def _should_use_colors() -> bool:
    """Checks if the environment supports colors and the user wants them."""
    if os.environ.get('NO_COLOR'):
        return False
    return sys.stdout.isatty()


_ = tqdm.get_lock()  # type: ignore[reportUnknownMemberType]


# --- Core I/O Operations ---


def _write_output[T: IOBase](
    *,
    operation: Literal['Downloading', 'Extracting'],
    input: Callable[[], AbstractContextManager[T]],
    input_size: Callable[[T], int | None],
    output: Path,
    quiet: bool,
    enable_colors: bool,
) -> Path:
    """Write input to output, updating a progress bar for the user when requested."""
    progress = tqdm(
        desc=f'{operation} {output.name}',
        disable=quiet,
        colour=next(_TAB10_COLORS) if enable_colors else None,
        unit='bytes',
        unit_scale=True,
    )
    try:
        with input() as input_file, open(output, 'wb') as output_file:
            progress.reset(total=input_size(input_file))

            CHUNK_SIZE: Final = 4096
            while chunk := input_file.read(CHUNK_SIZE):
                _ = output_file.write(chunk)
                _ = progress.update(len(chunk))

        return output

    except Exception as error:
        progress.display(f'{error}')
        output.unlink(missing_ok=True)
        raise error

    finally:
        progress.refresh()
        progress.close()


def _download_dump(dump_url: ParseResult, output_dir: Path, *, quiet: bool, enable_colors: bool) -> Path:
    """Download compressed dump file."""
    return _write_output(
        operation='Downloading',
        input=lambda: urlopen(dump_url.geturl()),
        input_size=lambda response: int(response.info().get('Content-Length', -1)),
        output=output_dir / Path(dump_url.path).name,
        quiet=quiet,
        enable_colors=enable_colors,
    )


def _get_gzip_uncompressed_size(gzip_path: Path) -> int | None:
    """
    Reads the last 4 bytes of a gzip file to get the uncompressed size.
    This should work for files under 4GB, as per the gzip format spec (ISIZE).
    """
    try:
        with open(gzip_path, 'rb') as file:
            _ = file.seek(-4, 2)
            return struct.unpack('<I', file.read(4))[0]
    except (struct.error, OSError):
        return None


def _extract_dump(dump_gz_path: Path, *, quiet: bool, enable_colors: bool) -> Path:
    """Uncompress downloaded dump file."""
    return _write_output(
        operation='Extracting',
        input=lambda: gzip.open(dump_gz_path, 'rb'),
        input_size=lambda _: _get_gzip_uncompressed_size(dump_gz_path),
        output=dump_gz_path.with_suffix(''),
        quiet=quiet,
        enable_colors=enable_colors,
    )


def _process_single_dump(dump_url: ParseResult, output_dir: Path, *, quiet: bool, enable_colors: bool) -> Path:
    """Download and extract a PCAP dump."""
    gzip_path = _download_dump(dump_url, output_dir, quiet=quiet, enable_colors=enable_colors)
    return _extract_dump(gzip_path, quiet=quiet, enable_colors=enable_colors)


# --- Data Source Parsing ---

# URLs to extract from data sources
_URL_PATTERN: Final = re.compile(r'\bhttp[s]?://([a-zA-Z0-9]+[.])+([a-zA-Z0-9_-]+[/])+[0-9]+\.dump\.gz\b')


def _resolve_data_sources(sources: Iterable[Path], *, recursive: bool = False) -> Iterator[Path]:
    """Markdown files with download URLs in them."""
    for source in sources:
        if source.is_dir():
            for file in glob.iglob('*.md', root_dir=source, recursive=recursive):
                yield source / file
        else:
            yield source


def _get_dump_urls(source_file: Path) -> Iterator[ParseResult]:
    """List markdown files with download URLs in them."""
    with open(source_file) as file:
        for line in file:
            for url_match in _URL_PATTERN.finditer(line):
                yield urlparse(url_match.group(0))


# --- Main Asynchronous Logic ---


@asynccontextmanager
async def _with_no_wait_task_group() -> AsyncIterator[asyncio.TaskGroup]:
    """Create a task group that exits immediately on exceptions."""
    with ThreadPoolExecutor() as executor:
        loop = asyncio.get_running_loop()
        loop.set_default_executor(executor)

        try:
            async with asyncio.TaskGroup() as task_group:
                yield task_group
        finally:
            executor.shutdown(wait=True, cancel_futures=False)


async def _process_all_dumps(sources: Iterable[Path], *, quiet: bool, enable_colors: bool = True) -> int:
    """Download and extract all dump files from all sources listed."""
    tasks: list[asyncio.Task[Path]] = []

    async with _with_no_wait_task_group() as task_group:
        for source in sources:
            output_dir = source.parent
            for url in _get_dump_urls(source):
                coro = asyncio.to_thread(
                    _process_single_dump,
                    url,
                    output_dir,
                    quiet=quiet,
                    enable_colors=enable_colors,
                )
                tasks.append(task_group.create_task(coro))

    error_count = 0
    for task in tasks:
        if (error := task.exception()) is not None:
            error_count += 1
            if not quiet:
                print_exception(error)

    return -error_count


def main() -> int:
    """Download and extract PCAP files from specified sources"""
    parser = ArgumentParser('download', description='Download and extract PCAP files from specified sources.')
    _ = parser.add_argument('source', nargs='+', type=Path, help='File or directory to search for data urls.')
    _ = parser.add_argument('-r', '--recursive', action='store_true', help='Recurse into the SOURCE directories.')
    _ = parser.add_argument('-q', '--quiet', action='store_true', help="Don't display progress.")
    _ = parser.add_argument('-c', '--color', action=BooleanOptionalAction, default=None, help='Display colored output.')

    args = parser.parse_intermixed_args()

    data_sources = _resolve_data_sources(args.source, recursive=args.recursive)
    if not data_sources:
        print('No valid source files found.', file=sys.stderr)
        return 1

    enable_colors = bool(args.color) if args.color is not None else _should_use_colors()
    if enable_colors:
        colored_traceback.add_hook()

    try:
        return asyncio.run(_process_all_dumps(data_sources, quiet=args.quiet, enable_colors=enable_colors))
    except KeyboardInterrupt:
        return SIGINT


if __name__ == '__main__':
    sys.exit(main())
