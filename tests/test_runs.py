"""Acceptance tests for the three reported demonstration runs."""

from runs.run1_ablation import assert_acceptance as assert_run1, evaluate as run1
from runs.run2_reconciliation import evaluate as run2
from runs.run3_faithfulness import assert_acceptance as assert_run3, evaluate as run3


def test_run1_acceptance() -> None:
    assert_run1(run1())


def test_run2_acceptance() -> None:
    minimum, average = run2()
    assert len(minimum.outcomes) == len(average.outcomes) == 20
    for simulation in (minimum, average):
        final = simulation.outcomes[-1]
        assert final.terminal_timestamp == final.rounds[-1].timestamp
    weakest = minimum.outcomes[-1]
    for edge_key in weakest.results["a3"].implicated:
        assert minimum.lewm.edges[edge_key].t_last == weakest.terminal_timestamp


def test_run3_acceptance() -> None:
    rows, _ = run3()
    assert_run3(rows)
