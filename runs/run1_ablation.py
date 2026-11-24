"""Run 1: source-discount/source-typing ablation on trusted testimony."""

import random
from typing import Dict, List

from tomu.config import DEFAULT_CONFIG, SimulationConfig
from tomu.inference import Mode, backward_inference
from tomu.scenarios import build_variant

from .common import aligned_table, write_csv, write_text


def evaluate(config: SimulationConfig = DEFAULT_CONFIG) -> List[Dict[str, object]]:
    """Evaluate all four discount/typing mode combinations."""

    rows: List[Dict[str, object]] = []
    modes = [Mode(True, True), Mode(True, False), Mode(False, True), Mode(False, False)]
    for mode in modes:
        scenario = build_variant("trusted", config)
        target = scenario.lewm.agents[scenario.target]
        result = backward_inference(
            target, scenario.behavior, "contents", mode,
            config.s1_default, config, focal=scenario.lewm.focal,
        )
        rows.append({
            "discounts": mode.apply_discounts,
            "typed": mode.typed,
            "attribution": result.value,
            "confidence": result.confidence,
            "classification": result.status,
            "diagonal": mode.apply_discounts == mode.typed,
        })
    return rows


def assert_acceptance(
    rows: List[Dict[str, object]], config: SimulationConfig = DEFAULT_CONFIG
) -> None:
    """Assert the two diagonal headline criteria (Appendix A.4.1, Table A3)."""

    by_mode = {(row["discounts"], row["typed"]): row for row in rows}
    full = by_mode[(True, True)]
    surface = by_mode[(False, False)]
    assert full["attribution"] == "popcorn"
    assert float(full["confidence"]) > config.tau_high
    assert surface["attribution"] == "chocolate"


def render(rows: List[Dict[str, object]]) -> str:
    """Render the measured 2x2 ablation with diagonal cells called out."""

    display = []
    for row in rows:
        display.append([
            "T" if row["discounts"] else "F",
            "T" if row["typed"] else "F",
            row["attribution"],
            f"{float(row['confidence']):.6f}",
            row["classification"],
            "*" if row["diagonal"] else "",
        ])
    title = "Run 1 — source-discount/source-typing ablation (* = headline diagonal)"
    return title + "\n\n" + aligned_table(
        ["discounts", "typed", "attribution", "confidence", "classification", "diagonal"],
        display,
    )


def main() -> None:
    """Execute Run 1, assert acceptance, and write TXT/CSV artifacts."""

    random.seed(DEFAULT_CONFIG.seed)
    rows = evaluate()
    assert_acceptance(rows)
    text = render(rows)
    write_text("run1_ablation.txt", text)
    csv_rows = [dict(row, confidence=f"{float(row['confidence']):.12f}") for row in rows]
    write_csv(
        "run1_ablation.csv", csv_rows,
        ["discounts", "typed", "attribution", "confidence", "classification", "diagonal"],
    )
    print(text)


if __name__ == "__main__":
    main()
