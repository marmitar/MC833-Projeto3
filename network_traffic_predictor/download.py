"""Download files listed in data sources."""

import asyncio
import glob
import gzip
import re
import struct
from collections.abc import Callable, Iterable, Iterator
from contextlib import AbstractContextManager
from io import IOBase
from pathlib import Path
from typing import Final
from urllib.parse import ParseResult, urlparse
from urllib.request import urlopen

from tqdm import tqdm

# points to "${project_root}/data"
_DATA_DIR: Final = Path(__file__).parent.resolve(strict=True)

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


def _data_sources() -> Iterator[Path]:
    """Markdown files with download URLs in them."""
    for file in glob.iglob('*.md', root_dir=_DATA_DIR):
        yield _DATA_DIR / file


def _get_dump_urls(source_file: Path) -> Iterator[ParseResult]:
    """Markdown files with download URLs in them."""
    with open(source_file) as file:
        for line in file:
            for url_match in _URL_PATTERN.finditer(line):
                yield urlparse(url_match.group(0))


def download_dump(dump_url: ParseResult) -> Path:
    """Download compressed dump file."""

    filename = Path(dump_url.path).name
    return _write_with_progress(
        description=f'Downloading {filename}',
        input=lambda: urlopen(dump_url.geturl()),
        input_size=lambda response: int(response.info().get('Content-Length', -1)),
        output=_DATA_DIR / filename,
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


async def _get_dump(dump_url: ParseResult) -> Path | None:
    try:
        gzip_path = await asyncio.to_thread(download_dump, dump_url)
        dump_path = await asyncio.to_thread(extract_dump, gzip_path)
        return dump_path
    except Exception:
        return None


async def _get_all_dumps(sources: Iterable[Path]) -> None:
    async with asyncio.TaskGroup() as tg:
        for source in sources:
            for url in _get_dump_urls(source):
                _ = tg.create_task(_get_dump(url))


def main():
    asyncio.run(_get_all_dumps(_data_sources()))


if __name__ == '__main__':
    main()
