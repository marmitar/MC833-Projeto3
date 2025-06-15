"""Download files listed in data sources."""

import glob
import gzip
import re
import shutil
from collections.abc import Callable, Iterator
from functools import wraps
from itertools import chain
from multiprocessing import Pool
from pathlib import Path
from typing import Annotated, Final
from urllib.parse import urlparse
from urllib.request import urlopen

# points to "${project_root}/data"
DATA_DIR: Final = Path(__file__).parent.resolve(strict=True)

# URLs to extract from data sources
URL_PATTERN: Final = re.compile(r'\bhttp[s]?://([a-zA-Z0-9]+[.])+([a-zA-Z0-9_-]+[/])+[0-9]+\.dump\.gz\b')


def collect[T, **P](generator: Callable[P, Iterator[T]]) -> Callable[P, tuple[T, ...]]:
    """Collect all the results of generator into tuple before returning."""

    @wraps(generator)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> tuple[T, ...]:
        return tuple(generator(*args, **kwargs))

    return wrapper


def data_sources() -> Iterator[Path]:
    """Markdown files with download URLs in them."""

    for file in glob.iglob('*.md', root_dir=DATA_DIR):
        yield DATA_DIR / file


type Url = Annotated[str, 'A valid URL']


@collect
def get_dump_urls(source_file: Path) -> Iterator[Url]:
    """Markdown files with download URLs in them."""

    with open(source_file) as file:
        for line in file:
            for url_match in URL_PATTERN.finditer(line):
                yield url_match.group(0)


def download_dump(dump_url: Url) -> Path | None:
    """Download compressed dump file."""

    print(f'Downloading {dump_url}...')
    filename = Path(urlparse(dump_url).path).name
    output = DATA_DIR / filename

    try:
        with urlopen(dump_url) as response, open(output, 'wb') as out_file:
            shutil.copyfileobj(response, out_file, length=8192)

        print(f'Downloaded {filename}.')
        return output

    except Exception as error:
        print(f'Error while downloading {dump_url}: {error}')
        output.unlink(missing_ok=True)
        return None


def extract_dump(dump_filename: Path) -> Path | None:
    """Uncompress downloaded dump file."""

    print(f'Extracting {dump_filename}...')
    gz_path = DATA_DIR / dump_filename
    output_filename = gz_path.with_suffix('')

    try:
        with gzip.open(gz_path, 'rb') as gz_file, open(output_filename, 'wb') as raw_file:
            shutil.copyfileobj(gz_file, raw_file)

        print(f'Extracted {dump_filename}.')
        return output_filename

    except Exception as error:
        print(f'Error while extracting {dump_filename}: {error}')
        output_filename.unlink(missing_ok=True)
        return None


def main():
    with Pool() as pool:
        urls = pool.imap_unordered(get_dump_urls, data_sources())
        dumps = pool.imap_unordered(download_dump, chain.from_iterable(urls))
        results = pool.imap_unordered(extract_dump, (dump for dump in dumps if dump))
        for result in results:
            print(f'Done {result}.')


if __name__ == '__main__':
    main()
