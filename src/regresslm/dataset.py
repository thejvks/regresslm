"""Load golden datasets from YAML or JSONL."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from .schema import Case, Dataset


def load_dataset(path: str | Path) -> Dataset:
    path = Path(path)
    if path.suffix in {".yaml", ".yml"}:
        raw = yaml.safe_load(path.read_text())
        name = raw.get("name", path.stem)
        cases = [Case(**c) for c in raw.get("cases", [])]
    elif path.suffix == ".jsonl":
        name = path.stem
        cases = [Case(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
    else:
        raise ValueError(f"unsupported dataset format: {path.suffix}")
    return Dataset(name=name, cases=cases)
