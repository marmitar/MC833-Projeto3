import glob
import gzip
import re
import shutil
from collections.abc import Iterator
from itertools import chain
from multiprocessing import Pool
from os import PathLike
from pathlib import Path
from typing import Final
from urllib.parse import urlparse
from urllib.request import urlopen

DATA_DIR: Final = Path(__file__).parent.resolve(strict=True)

URL_PATTERN: Final = re.compile(r'http[s]?://([a-zA-Z0-9]+[.])+([a-zA-Z0-9_-]+[/])+[0-9]+\.dump\.gz')


def data_sources() -> Iterator[Path]:
    for file in glob.iglob('*.md', root_dir=DATA_DIR):
        yield DATA_DIR / file


def get_dump_urls(source_file: str | PathLike[str]) -> tuple[str, ...]:
    with open(source_file) as file:
        urls = tuple(url.group(0) for line in file for url in URL_PATTERN.finditer(line))
    return urls


def download_dump(dump_url: str) -> Path | None:
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


def extract_dump(dump_filename: str | PathLike[str]) -> Path | None:
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
    with Pool() as p:
        urls = p.imap_unordered(get_dump_urls, data_sources())
        dumps = p.imap_unordered(download_dump, chain.from_iterable(urls))
        results = p.imap_unordered(extract_dump, (dump for dump in dumps if dump))
        for result in results:
            print(f'Done {result}.')


if __name__ == '__main__':
    main()
