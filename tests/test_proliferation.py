"""Tests for bounded branching and concrete projected nodes."""

from dataclasses import replace

from tomu.agent_node import ProjectedAgentNode
from tomu.config import DEFAULT_CONFIG
from tomu.inference import mutual_reconciliation
from tomu.proliferation import branching_width, rebuild_and_prune
from tomu.scenarios import build_depth2_friend, build_run2


def test_branching_width_decays() -> None:
    assert branching_width(1, 3, 0.5) == 1
    assert branching_width(2, 3, 0.5) == 0


def test_low_confidence_never_prunes_mandatory_depth_one() -> None:
    lewm, targets, _ = build_run2(DEFAULT_CONFIG)
    confidences = {target: 0.2 for target in targets}
    branches = rebuild_and_prune(lewm, targets, confidences, 3, DEFAULT_CONFIG)
    assert [(branch.target, branch.depth) for branch in branches] == [
        ("a2", 1), ("a3", 1), ("a4", 1)
    ]


def _friend_outcome(projected_claim: str):
    config = replace(DEFAULT_CONFIG, rho=0.8, tau_high=0.99)
    scenario, templates = build_depth2_friend(config, projected_claim)
    outcome = mutual_reconciliation(
        scenario.lewm,
        [scenario.target],
        {scenario.target: scenario.behavior},
        "contents",
        config.s1_default,
        "min",
        1.0,
        config,
        projection_templates=templates,
    )
    return scenario, outcome


def test_depth_two_friend_is_a_concrete_projected_agent() -> None:
    scenario, outcome = _friend_outcome("popcorn")
    branch = next(
        branch for branch in outcome.rounds[0].branches if branch.depth == 2
    )
    node = scenario.lewm.agents[branch.node]
    assert isinstance(node, ProjectedAgentNode)
    assert node.parent == "a2"
    assert node.path == ("a2", "friend")
    assert node.represented_agent == "friend"
    assert node.history[0].claim == "popcorn"
    assert outcome.rounds[0].branch_results[branch.node].value == "popcorn"


def test_depth_two_friend_content_changes_joint_inference() -> None:
    _, coherent = _friend_outcome("popcorn")
    _, perturbed = _friend_outcome("chocolate")
    assert coherent.confidence > perturbed.confidence
    assert coherent.status == "ACCEPTED_LOW"
    assert perturbed.status == "REJECTED"


def test_target_sophistication_caps_recursive_depth() -> None:
    config = replace(DEFAULT_CONFIG, rho=0.8)
    scenario, templates = build_depth2_friend(config, "popcorn")
    scenario.lewm.agents[scenario.target].sophistication = 1

    branches = rebuild_and_prune(
        scenario.lewm,
        [scenario.target],
        {scenario.target: 1.0},
        config.s1_default,
        config,
        templates,
    )

    assert [(branch.target, branch.depth) for branch in branches] == [
        ("a2", 1)
    ]