"""Executable structural checks for inference and proliferation composition."""

from dataclasses import replace

from tomu.agent_node import ProjectedAgentNode, Source
from tomu.config import DEFAULT_CONFIG
from tomu.inference import mutual_reconciliation, self_projection
from tomu.scenarios import build_depth2_friend, build_variant


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


def evaluate_checks():
    """Return named structural checks without mutating reported run artifacts."""

    scenario = build_variant("trusted", DEFAULT_CONFIG)
    scenario.lewm.agents[scenario.lewm.focal].history = [
        Source(
            "observer_signal", "chocolate", "contents", "belief",
            0.0, 1.0, 1.0, [],
        )
    ]
    scenario.lewm.state_nodes.append("observer_signal")
    projection = self_projection(
        scenario.lewm,
        scenario.target,
        "contents",
        DEFAULT_CONFIG.s1_default,
        DEFAULT_CONFIG,
        1.0,
    )

    coherent_scenario, coherent = _friend_outcome("popcorn")
    _, perturbed = _friend_outcome("chocolate")
    branch = next(
        branch for branch in coherent.rounds[0].branches if branch.depth == 2
    )
    node = coherent_scenario.lewm.agents[branch.node]
    return [
        ("SP selects from projected H1", projection.value == "chocolate"),
        ("SP content is distinct from BI content", projection.value != "popcorn"),
        ("depth-2 node is concrete", isinstance(node, ProjectedAgentNode)),
        ("depth-2 parent/path recorded", node.parent == "a2" and node.path == ("a2", "friend")),
        ("depth-2 content affects inference", coherent.confidence > perturbed.confidence),
    ]


def main() -> None:
    """Print deterministic PASS/FAIL rows and fail if any check is false."""

    checks = evaluate_checks()
    width = max(len(name) for name, _ in checks)
    for name, passed in checks:
        print(f"{name:<{width}}  {'PASS' if passed else 'FAIL'}")
    assert all(passed for _, passed in checks)


if __name__ == "__main__":
    main()
