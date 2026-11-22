"""
paths.py -- the project's folders, so that every script reads and writes data in the
same places, whichever folder it is started from.

    cubespline/
      code/            the Python packages (core, optimization, grid, visualization, tests)
      examples/        example targets for the grid: functions/*.py, point_clouds/*
      data/fits/       fit files written by the optimizers (default --out); grid fits in data/fits/grid
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CODE = REPO / "code"
DATA = REPO / "data"
FITS = DATA / "fits"
GRID_FITS = FITS / "grid"               # fits written by the grid optimizer
EXAMPLES = REPO / "examples"            # example targets: functions/*.py, point_clouds/*
FIT_FOLDERS = (FITS, GRID_FITS)          # where the viewers look for fit files


def output_path(name: str, folder: Path = FITS) -> str:
    """Default location for an output file: `folder / name` (the folder is created)."""
    folder.mkdir(parents=True, exist_ok=True)
    return str(folder / name)


def find_input(path: str) -> str:
    """A file given on the command line: as given (relative to the current folder) if it
    exists, else looked up by name in data/fits and data/fits/grid."""
    p = Path(path).expanduser()
    if p.exists():
        return str(p)
    for folder in FIT_FOLDERS:
        if (folder / p).exists():
            return str(folder / p)
    return str(p)
