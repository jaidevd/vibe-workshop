"""Download and load IMDb datasets.

The workshop environment ships with the IMDb datasets already extracted in
some scenarios.  To avoid redundant downloads we first attempt to locate both
the compressed (``.tsv.gz``) and uncompressed (``.tsv``) variants before
falling back to a network fetch.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd
import requests

# IMDb dataset download URLs. The datasets are public and updated daily.
DATASET_URLS: Dict[str, str] = {
    "title.basics.tsv.gz": "https://datasets.imdbws.com/title.basics.tsv.gz",
    "title.ratings.tsv.gz": "https://datasets.imdbws.com/title.ratings.tsv.gz",
}

# For each dataset track alternative on-disk representations we are willing to
# consume.  The keys match :data:`DATASET_URLS` but we also include the
# decompressed ``.tsv`` payload that may already exist on disk.
DATASET_ALIASES: Dict[str, Iterable[str]] = {
    name: (name, name[:-3]) if name.endswith(".gz") else (name,)
    for name in DATASET_URLS
}

logger = logging.getLogger(__name__)


def _candidate_directories(data_dir: Path) -> Iterable[Path]:
    """Return directories that might already host the datasets.

    The user can provide a ``--data-dir`` flag, set the ``IMDB_DATA_DIR``
    environment variable, or rely on a conventional ``~/data`` folder.  The
    helper yields each location exactly once and silently skips entries that do
    not exist.
    """

    env_dir = os.environ.get("IMDB_DATA_DIR")
    candidates = [data_dir]
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.append(Path.home() / "data")

    seen = set()
    for raw_dir in candidates:
        try:
            directory = raw_dir.resolve()
        except FileNotFoundError:
            directory = raw_dir
        if directory in seen:
            continue
        seen.add(directory)
        yield directory


def _find_existing_dataset(name: str, data_dir: Path) -> Optional[Path]:
    """Return the path of ``name`` if it already exists on disk."""

    aliases = DATASET_ALIASES.get(name, (name,))
    for base_dir in _candidate_directories(data_dir):
        for alias in aliases:
            candidate = base_dir / alias
            if candidate.exists():
                logger.info("Using local dataset %s", candidate)
                return candidate

        # Also inspect one level of sub-directories to catch layouts such as
        # ``<data_dir>/imdb/title.basics.tsv`` without walking arbitrarily deep
        # directory trees.
        if base_dir.exists():
            for child in base_dir.iterdir():
                if not child.is_dir():
                    continue
                for alias in aliases:
                    candidate = child / alias
                    if candidate.exists():
                        logger.info("Using local dataset %s", candidate)
                        return candidate
    return None


def download_dataset(name: str, data_dir: Path) -> Path:
    """Ensure that the requested dataset exists locally.

    Parameters
    ----------
    name:
        Filename in :data:`DATASET_URLS`.
    data_dir:
        Directory where the dataset should be stored.

    Returns
    -------
    Path
        Location of the downloaded dataset.
    """

    if name not in DATASET_URLS:
        raise KeyError(f"Unknown dataset: {name}")

    existing = _find_existing_dataset(name, data_dir)
    if existing is not None:
        return existing

    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / name

    if path.exists():
        logger.info("Dataset %s already present at %s", name, path)
        return path

    url = DATASET_URLS[name]
    logger.info("Downloading %s to %s", url, path)

    with requests.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()
        with path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1 << 15):
                if chunk:
                    f.write(chunk)

    logger.info("Finished downloading %s", name)
    return path


def ensure_datasets(data_dir: Path) -> Dict[str, Path]:
    """Download the IMDb datasets needed for the analysis."""

    paths = {name: download_dataset(name, data_dir) for name in DATASET_URLS}
    return paths


def load_ratings(path: Path) -> pd.DataFrame:
    """Load title ratings into a :class:`~pandas.DataFrame`."""

    ratings = pd.read_csv(
        path,
        sep="\t",
        compression="infer",
        dtype={"tconst": "string"},
        na_values="\\N",
    )
    ratings["numVotes"] = ratings["numVotes"].astype("int64")
    return ratings
