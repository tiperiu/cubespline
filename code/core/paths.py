"""
paths.py -- the project's folders, so that every script reads and writes data in the
same places, whichever folder it is started from.

    cubespline/
      code/            the Python packages (core, optimization, visualization, benchmark, tests)
      data/fits/       fit files written by the optimizers (default --out)
      data/benchmark/  benchmark results: benchmark_warp.json, saved fits, overview images
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CODE = REPO / "code"
DATA = REPO / "data"
FITS = DATA / "fits"
BENCHMARK = DATA / "benchmark"
FIT_FOLDERS = (FITS, BENCHMARK)          # where the viewers look for fit files


def output_path(name: str, folder: Path = FITS) -> str:
    """Default location for an output file: `folder / name` (the folder is created)."""
    folder.mkdir(parents=True, exist_ok=True)
    return str(folder / name)


def find_input(path: str) -> str:
    """A file given on the command line: as given (relative to the current folder) if it
    exists, else looked up by name in data/fits and data/benchmark."""
    p = Path(path).expanduser()
    if p.exists():
        return str(p)
    for folder in FIT_FOLDERS:
        if (folder / p).exists():
            return str(folder / p)
    return str(p)
