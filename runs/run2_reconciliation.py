"""Run 2: reconciliation rule changes concentrated versus spread residue (Appendix A.4.2)."""

import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from tomu.config import DEFAULT_CONFIG, SimulationConfig
from tomu.inference import Outcome, mutual_reconciliation
from tomu.lewm import LEWM
from tomu.residue import apply_to_edges, floored_edges, sync_source_credibilities
from tomu.scenarios import build_run2

from .common import OUTPUT_DIR, aligned_table, write_csv, write_text


@dataclass
class Simulation:
    """One termination rule's final LEWM, outcomes, and edge time series."""

    termination: str
    lewm: LEWM
    outcomes: List[Outcome]
    history: List[Dict[str, object]]


def _apply_rejected_outcome(
    lewm: LEWM,
    outcome: Outcome,
    termination: str,
    config: SimulationConfig,
) -> None:
    """Apply residue at the terminal reconciliation timestamp."""

    if outcome.status != "REJECTED":
        return
    if termination == "min":
        weakest = min(sorted(outcome.pair), key=lambda key: outcome.pair[key])
        disconfirmation = 1.0 - outcome.pair[weakest]
        apply_to_edges(
            lewm,
            outcome.results[weakest].implicated,
            disconfirmation,
            outcome.terminal_timestamp,
            config,
        )
        return
    for target in sorted(outcome.pair):
        if outcome.pair[target] < 1.0:
            disconfirmation = 1.0 - outcome.pair[target]
            apply_to_edges(
                lewm,
                outcome.results[target].implicated,
                disconfirmation,
                outcome.terminal_timestamp,
                config,
            )


def _record_edges(
    lewm: LEWM, termination: str, invocation: int
) -> List[Dict[str, object]]:
    """Snapshot all edge credibility/floor values after one invocation."""

    records: List[Dict[str, object]] = []
    for key in sorted(lewm.edges):
        edge = lewm.edges[key]
        if not edge.report:
            continue
        records.append({
            "termination": termination,
            "invocation": invocation,
            "edge": f"{key[0]}->{key[1]}",
            "cred": edge.cred,
            "floor": edge.floor,
        })
    return records


def simulate(
    termination: str,
    config: SimulationConfig = DEFAULT_CONFIG,
    invocations: int = -1,
) -> Simulation:
    """Run repeated reconciliation while carrying residue forward (Appendix A.4.2)."""

    lewm, targets, behaviors = build_run2(config)
    count = config.run2_invocations if invocations < 0 else invocations
    outcomes: List[Outcome] = []
    history: List[Dict[str, object]] = []
    for invocation in range(1, count + 1):
        sync_source_credibilities(lewm)
        outcome = mutual_reconciliation(
            lewm, targets, behaviors, "contents", config.s1_default,
            termination, float(invocation), config,
        )
        outcomes.append(outcome)
        _apply_rejected_outcome(lewm, outcome, termination, config)
        history.extend(_record_edges(lewm, termination, invocation))
    return Simulation(termination, lewm, outcomes, history)


def weakest_pair_share(lewm: LEWM, weakest: str = "a3") -> float:
    """Return weakest target's fraction of total persistent floor (Appendix A.4.2, Table A4)."""

    total = sum(edge.floor for edge in lewm.edges.values())
    weakest_total = 0.0
    for key, edge in lewm.edges.items():
        if key == (lewm.focal, weakest) or key[0] == weakest:
            weakest_total += edge.floor
    return weakest_total / total if total else 0.0


def metrics(simulation: Simulation) -> Tuple[int, float]:
    """Return floored-edge count and weakest-pair floor share."""

    return len(floored_edges(simulation.lewm)), weakest_pair_share(simulation.lewm)


def assert_acceptance(minimum: Simulation, average: Simulation) -> None:
    """Assert the two final residue-divergence inequalities (Appendix A.4.2)."""

    min_count, min_share = metrics(minimum)
    avg_count, avg_share = metrics(average)
    assert min_count < avg_count
    assert min_share > avg_share
    assert all(outcome.status == "REJECTED" for outcome in minimum.outcomes)
    assert all(outcome.status == "REJECTED" for outcome in average.outcomes)


def evaluate(config: SimulationConfig = DEFAULT_CONFIG) -> Tuple[Simulation, Simulation]:
    """Run and validate both mutual-reconciliation termination rules."""

    minimum = simulate("min", config)
    average = simulate("weighted_average", config)
    assert_acceptance(minimum, average)
    return minimum, average


def _render(minimum: Simulation, average: Simulation) -> str:
    """Render final metrics and per-edge residue state for both rules."""

    summary = []
    edge_rows = []
    for simulation in [minimum, average]:
        count, share = metrics(simulation)
        summary.append([simulation.termination, count, f"{share:.6f}"])
        for key in sorted(simulation.lewm.edges):
            edge = simulation.lewm.edges[key]
            if not edge.report:
                continue
            edge_rows.append([
                simulation.termination, f"{key[0]}->{key[1]}",
                f"{edge.cred:.6f}", f"{edge.floor:.6f}",
            ])
    return "\n\n".join([
        "Run 2 — reconciliation residue divergence",
        aligned_table(["termination", "n_floored", "weakest_pair_share"], summary),
        aligned_table(["termination", "edge", "cred", "floor"], edge_rows),
    ])


def _plot_termination(axes: Any, history: List[Dict[str, object]], termination: str) -> None:
    """Add three focal estimate-edge series for one termination rule."""

    style = "--" if termination == "min" else "-"
    markers = {"a2": "o", "a3": "s", "a4": "^"}
    for target in ["a2", "a3", "a4"]:
        edge_name = f"a1->{target}"
        series = [
            row for row in history
            if row["termination"] == termination and row["edge"] == edge_name
        ]
        label = f"{termination}: {target}"
        x_values = [row["invocation"] for row in series]
        plot_args = {
            "color": "black",
            "linestyle": style,
            "marker": markers[target],
            "markevery": 3,
            "markersize": 3.5,
            "linewidth": 1.1,
            "label": label,
        }
        axes[0].plot(x_values, [row["cred"] for row in series], **plot_args)
        axes[1].plot(x_values, [row["floor"] for row in series], **plot_args)


def _plot(history: List[Dict[str, object]]) -> None:
    """Optionally render credibility/floor time series as a two-panel PDF."""

    cache_dir = OUTPUT_DIR.parent / ".matplotlib-cache"
    try:
        cache_dir.mkdir(exist_ok=True)
    except OSError:
        cache_dir = OUTPUT_DIR / "matplotlib-cache"
        cache_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; Run 2 PDF figure skipped.")
        return
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
    })
    OUTPUT_DIR.mkdir(exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 4.1))
    for termination in ["min", "weighted_average"]:
        _plot_termination(axes, history, termination)
    axes[0].set(title="Focal estimate credibility", xlabel="invocation", ylabel="cred")
    axes[1].set(title="Focal estimate persistence floor", xlabel="invocation", ylabel="floor")
    for axis in axes:
        axis.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    labels = [
        "min: a2", "weighted_average: a2",
        "min: a3", "weighted_average: a3",
        "min: a4", "weighted_average: a4",
    ]
    handles = [by_label[label] for label in labels]
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=True)
    figure.tight_layout(rect=(0.0, 0.17, 1.0, 1.0))
    fixed_time = datetime(2000, 1, 1, tzinfo=timezone.utc)
    figure.savefig(
        OUTPUT_DIR / "run2_reconciliation.pdf",
        metadata={"CreationDate": fixed_time, "ModDate": fixed_time},
    )
    plt.close(figure)


def main() -> None:
    """Execute Run 2, assert acceptance, and write TXT/CSV/PDF artifacts."""

    random.seed(DEFAULT_CONFIG.seed)
    minimum, average = evaluate()
    history = minimum.history + average.history
    text = _render(minimum, average)
    write_text("run2_reconciliation.txt", text)
    csv_rows = [dict(row, cred=f"{float(row['cred']):.12f}", floor=f"{float(row['floor']):.12f}") for row in history]
    write_csv("run2_reconciliation.csv", csv_rows, ["termination", "invocation", "edge", "cred", "floor"])
    _plot(history)
    print(text)


if __name__ == "__main__":
    main()
