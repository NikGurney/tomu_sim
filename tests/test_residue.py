"""Unit tests for residue mutation, decay, and targeting."""

from checks.verify_residue import evaluate_checks
from tomu.config import DEFAULT_CONFIG
from tomu.residue import implicated_edges
from tomu.scenarios import build_variant


def test_all_boundary_conditions_pass() -> None:
    assert all(passed for _, passed in evaluate_checks())


def test_implicated_edges_are_estimate_plus_sources() -> None:
    scenario = build_variant("trusted", DEFAULT_CONFIG)
    target = scenario.lewm.agents[scenario.target]
    keys = implicated_edges(scenario.lewm.focal, target, target.history, "contents")
    assert keys == {("a1", "a2"), ("a2", "label"), ("a2", "testimony")}
