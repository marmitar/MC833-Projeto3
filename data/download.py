"""Download files listed in data sources."""

import glob
import gzip
import re
import struct
from collections.abc import Callable, Iterable, Iterator
from contextlib import AbstractContextManager
from functools import wraps
from io import IOBase
from itertools import chain, count
from multiprocessing import Pool
from pathlib import Path
from typing import Annotated, Final, NoReturn
from urllib.parse import urlparse
from urllib.request import urlopen

from tqdm import tqdm

shared_tqdm: type['tqdm[NoReturn]'] = tqdm


def set_shared_tqdm(tqdm: type['tqdm[NoReturn]']) -> None:
    """Setup shared classes state for a given process."""
    global shared_tqdm
    shared_tqdm = tqdm


def init_shared_tqdm_lock() -> object:
    """Initialize shared state across processes."""
    return tqdm.get_lock()  # type: ignore[reportUnknownMemberType]


# points to "${project_root}/data"
DATA_DIR: Final = Path(__file__).parent.resolve(strict=True)

# URLs to extract from data sources
URL_PATTERN: Final = re.compile(r'\bhttp[s]?://([a-zA-Z0-9]+[.])+([a-zA-Z0-9_-]+[/])+[0-9]+\.dump\.gz\b')


def collecting_iterator[T, **P](generator: Callable[P, Iterator[T]]) -> Callable[P, tuple[T, ...]]:
    """Collect all the results of generator into tuple before returning."""

    @wraps(generator)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> tuple[T, ...]:
        return tuple(generator(*args, **kwargs))

    return wrapper


def write_with_progress[T: IOBase](
    *,
    description: str,
    position: int,
    input: Callable[[], AbstractContextManager[T]],
    input_size: Callable[[T], int | None],
    output: Path,
    chunk_size: int = 8192,
) -> Path | None:
    """Write input to output, showing updating a progress bar for the user."""

    progress = tqdm(desc=description, unit='bytes', unit_scale=True, position=position)
    try:
        with input() as input_file, open(output, 'wb') as output_file:
            file_size = input_size(input_file)
            progress.reset(total=file_size)

            while chunk := input_file.read(chunk_size):
                output_file.write(chunk)
                progress.update(len(chunk))

        return output
    except Exception as error:
        progress.display(f'{error}')

        output.unlink(missing_ok=True)
        return None
    finally:
        progress.refresh()
        progress.close()


def data_sources() -> Iterator[Path]:
    """Markdown files with download URLs in them."""

    for file in glob.iglob('*.md', root_dir=DATA_DIR):
        yield DATA_DIR / file


type Url = Annotated[str, 'A valid URL']


@collecting_iterator
def get_dump_urls(source_file: Path) -> Iterator[Url]:
    """Markdown files with download URLs in them."""

    with open(source_file) as file:
        for line in file:
            for url_match in URL_PATTERN.finditer(line):
                yield url_match.group(0)


def download_dump(args: tuple[int, Url]) -> Path | None:
    """Download compressed dump file."""

    position, dump_url = args
    filename = Path(urlparse(dump_url).path).name

    return write_with_progress(
        description=f'Downloading {filename}',
        position=position,
        input=lambda: urlopen(dump_url),
        input_size=lambda response: int(response.info().get('Content-Length', -1)),
        output=DATA_DIR / filename,
    )


def get_gzip_uncompressed_size(gzip_path: Path) -> int | None:
    """
    Reads the last 4 bytes of a gzip file to get the uncompressed size.
    This should work for files under 4GB, as per the gzip format spec (ISIZE).
    """
    try:
        with open(gzip_path, 'rb') as file:
            file.seek(-4, 2)
            return struct.unpack('<I', file.read(4))[0]
    except (struct.error, OSError):
        return None


def extract_dump(args: tuple[int, Path]) -> Path | None:
    """Uncompress downloaded dump file."""

    position, dump_gz_path = args
    output = dump_gz_path.with_suffix('')

    return write_with_progress(
        description=f'Extracting {output.name}',
        position=position,
        input=lambda: gzip.open(dump_gz_path, 'rb'),
        input_size=lambda _: get_gzip_uncompressed_size(dump_gz_path),
        output=output,
    )


GLOBAL_COUNTER = count()


def global_enumerate[T](items: Iterable[T]) -> Iterator[tuple[int, T]]:
    """Like enumerate, but indices are globally shared and never repeated."""
    return zip(GLOBAL_COUNTER, items, strict=False)


def main():
    init_shared_tqdm_lock()

    with Pool(initializer=set_shared_tqdm, initargs=(tqdm,)) as pool:
        urls = pool.imap_unordered(get_dump_urls, data_sources())
        dumps = pool.imap_unordered(download_dump, global_enumerate(chain.from_iterable(urls)))
        results = pool.imap_unordered(extract_dump, global_enumerate(dump for dump in dumps if dump))
        for _ in results:
            pass


if __name__ == '__main__':
    main()
