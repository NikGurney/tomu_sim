"""Shared deterministic text/CSV output helpers for the Appendix A.4 runs."""

import csv
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def aligned_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    """Render rows as a stable aligned plain-text table."""

    string_rows = [[str(value) for value in row] for row in rows]
    widths: List[int] = []
    for index, header in enumerate(headers):
        values = [len(row[index]) for row in string_rows]
        widths.append(max([len(header)] + values))
    header_line = "  ".join(header.ljust(widths[i]) for i, header in enumerate(headers))
    rule = "  ".join("-" * width for width in widths)
    body = [
        "  ".join(value.ljust(widths[i]) for i, value in enumerate(row))
        for row in string_rows
    ]
    return "\n".join([header_line, rule] + body)


def write_text(filename: str, content: str) -> None:
    """Write a UTF-8 text artifact beneath outputs/ with Unix newlines on every OS."""

    OUTPUT_DIR.mkdir(exist_ok=True)
    with (OUTPUT_DIR / filename).open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content.rstrip() + "\n")


def write_csv(filename: str, rows: List[Mapping[str, object]], fields: Sequence[str]) -> None:
    """Write deterministic CSV with explicit field order and Unix newlines."""

    OUTPUT_DIR.mkdir(exist_ok=True)
    with (OUTPUT_DIR / filename).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
