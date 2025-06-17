"""Download files listed in data sources."""

import asyncio
import glob
import gzip
import re
import struct
from argparse import ArgumentParser
from collections.abc import Callable, Iterable, Iterator
from contextlib import AbstractContextManager
from io import IOBase
from pathlib import Path
from typing import Final
from urllib.parse import ParseResult, urlparse
from urllib.request import urlopen

from tqdm import tqdm

# URLs to extract from data sources
_URL_PATTERN: Final = re.compile(r'\bhttp[s]?://([a-zA-Z0-9]+[.])+([a-zA-Z0-9_-]+[/])+[0-9]+\.dump\.gz\b')


def _write_with_progress[T: IOBase](
    *,
    description: str,
    input: Callable[[], AbstractContextManager[T]],
    input_size: Callable[[T], int | None],
    output: Path,
    chunk_size: int = 8192,
) -> Path:
    """Write input to output, showing updating a progress bar for the user."""
    progress = tqdm(desc=description, unit='bytes', unit_scale=True)
    try:
        with input() as input_file, open(output, 'wb') as output_file:
            file_size = input_size(input_file)
            progress.reset(total=file_size)

            while chunk := input_file.read(chunk_size):
                _ = output_file.write(chunk)
                _ = progress.update(len(chunk))

        return output
    except Exception as error:
        progress.display(f'{error}')
        output.unlink(missing_ok=True)
        raise
    finally:
        progress.refresh()
        progress.close()


def download_dump(dump_url: ParseResult, output_dir: Path) -> Path:
    """Download compressed dump file."""
    filename = Path(dump_url.path).name
    return _write_with_progress(
        description=f'Downloading {filename}',
        input=lambda: urlopen(dump_url.geturl()),
        input_size=lambda response: int(response.info().get('Content-Length', -1)),
        output=output_dir / filename,
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


def extract_dump(dump_gz_path: Path) -> Path:
    """Uncompress downloaded dump file."""
    output = dump_gz_path.with_suffix('')
    return _write_with_progress(
        description=f'Extracting {output.name}',
        input=lambda: gzip.open(dump_gz_path, 'rb'),
        input_size=lambda _: _get_gzip_uncompressed_size(dump_gz_path),
        output=output,
    )


def _resolve_data_sources(sources: Iterable[Path], *, recursive: bool = False) -> Iterator[Path]:
    """Markdown files with download URLs in them."""
    for source in sources:
        if source.is_dir():
            for file in glob.iglob('*.md', root_dir=source, recursive=recursive):
                yield source / file
        else:
            yield source


def _get_dump_urls(source_file: Path) -> Iterator[ParseResult]:
    """Markdown files with download URLs in them."""
    with open(source_file) as file:
        for line in file:
            for url_match in _URL_PATTERN.finditer(line):
                yield urlparse(url_match.group(0))


async def _get_dump(dump_url: ParseResult, output_dir: Path) -> Path:
    gzip_path = await asyncio.to_thread(download_dump, dump_url, output_dir)
    dump_path = await asyncio.to_thread(extract_dump, gzip_path)
    return dump_path


async def _get_all_dumps(sources: Iterable[Path]) -> None:
    tasks: list[asyncio.Task[Path]] = []

    async with asyncio.TaskGroup() as tg:
        for source in sources:
            output_dir = source.parent
            for url in _get_dump_urls(source):
                task = tg.create_task(_get_dump(url, output_dir))
                tasks.append(task)

    for task in tasks:
        _ = task.result()


def main():
    parser = ArgumentParser('download', description='Download and extract PCAP files from specified sources.')
    _ = parser.add_argument('source', nargs='+', type=Path, help='File or directory to search for data urls.')
    _ = parser.add_argument('-r', '--recursive', action='store_true', help='Recurse into the SOURCE directories.')

    args = parser.parse_intermixed_args()

    data_sources = _resolve_data_sources(args.source, recursive=args.recursive)
    asyncio.run(_get_all_dumps(data_sources))


if __name__ == '__main__':
    main()
