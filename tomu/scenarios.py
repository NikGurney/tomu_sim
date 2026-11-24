"""Deterministic worked-example and reconciliation scenarios."""

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .agent_node import AgentNode, Source
from .config import CONTENT_NODE_BY_VALUE, SimulationConfig
from .lewm import Edge, LEWM
from .proliferation import ProjectionTemplate


@dataclass(frozen=True)
class Scenario:
    """One worked-example LEWM and its observed behavior."""

    name: str
    lewm: LEWM
    target: str
    behavior: str
    motivated_prior: Optional[Tuple[str, str]] = None


def label_source() -> Source:
    """Create the salient chocolate label source (manuscript §4)."""

    return Source("label", "chocolate", "contents", "belief", 0.0, 1.0, 1.0, [])


def testimony_source() -> Source:
    """Create the trusted popcorn testimony source (manuscript §4)."""

    return Source(
        "testimony", "popcorn", "contents", "trust", 0.9, 0.85, 1.0,
        ["label"], "friend"
    )


def _variant_inputs(name: str) -> tuple:
    """Return ordered history and behavior for a variant."""

    if name == "standard":
        return [label_source()], "CONTENT"
    if name == "trusted":
        return [testimony_source(), label_source()], "CONTENT"
    if name == "naive":
        return [], "NEUTRAL"
    if name == "motivated":
        return [testimony_source(), label_source()], "CONTENT"
    raise ValueError(f"unknown scenario variant: {name}")


def _edges_for_history(
    focal: str,
    target: str,
    history: List[Source],
    config: SimulationConfig,
) -> Dict[tuple, Edge]:
    """Create estimate/source LEWM edges corresponding to one history."""

    edges = {
        (focal, target): Edge(focal, target, ("modeling", 1.0), 1.0, 1.0)
    }
    for source in history:
        scalar = source.trust if source.kind == "trust" else config.belief_source_strength
        edges[(target, source.key)] = Edge(
            target, source.key, (source.kind, scalar), source.obs, source.cred
        )
    return edges


def build_variant(name: str, config: SimulationConfig) -> Scenario:
    """Build one of the four fully specified popcorn-bag variants (manuscript §4, Appendix A.4.3)."""

    history, behavior = _variant_inputs(name)
    focal, target = "a1", "a2"
    agents = {
        focal: AgentNode(focal, [], config.s1_default, 0.0),
        target: AgentNode(target, history, config.s1_default, 0.0),
    }
    state_nodes = ["contents"] + sorted(CONTENT_NODE_BY_VALUE.values())
    state_nodes.extend(source.key for source in history)
    edges = _edges_for_history(focal, target, history, config)
    motivated_prior = None
    if name == "motivated":
        motivated_prior = (target, CONTENT_NODE_BY_VALUE["pickle"])
        edges[motivated_prior] = Edge(
            target,
            CONTENT_NODE_BY_VALUE["pickle"],
            ("belief", 1.0),
            1.0,
            1.0,
            report=False,
        )
    lewm = LEWM(focal, agents, state_nodes, edges)
    lewm.validate()
    return Scenario(name, lewm, target, behavior, motivated_prior)


def build_all_variants(config: SimulationConfig) -> List[Scenario]:
    """Return all worked-example variants in the published order."""

    names = ["standard", "trusted", "naive", "motivated"]
    return [build_variant(name, config) for name in names]


def _run2_source(target: str) -> Source:
    """Create one behavior-consistent concrete source for a Run 2 target."""

    claims = {"a2": "popcorn", "a3": "chocolate", "a4": "pickle"}
    return Source(f"{target}_source", claims[target], "contents", "trust", 0.95, 0.95, 1.0, [])


def build_run2(config: SimulationConfig) -> tuple:
    """Build the three-target mutual-reconciliation setup (manuscript Appendix A.4.2)."""

    focal = "a1"
    targets = sorted(config.run2_target_obs)
    agents = {focal: AgentNode(focal, [], config.s1_default, 0.0)}
    edges: Dict[tuple, Edge] = {}
    behaviors: Dict[str, str] = {}
    for target in targets:
        source = _run2_source(target)
        agents[target] = AgentNode(target, [source], config.s1_default, 0.0)
        edges[(focal, target)] = Edge(
            focal, target, ("modeling", 1.0), config.run2_target_obs[target], 1.0
        )
        edges[(target, source.key)] = Edge(
            target, source.key, (source.kind, source.trust), source.obs, 1.0
        )
        behaviors[target] = "CONTENT"
    state_nodes = ["contents"] + sorted(CONTENT_NODE_BY_VALUE.values())
    state_nodes.extend(f"{target}_source" for target in targets)
    lewm = LEWM(focal, agents, state_nodes, edges)
    lewm.validate()
    return lewm, targets, behaviors


def build_depth2_friend(
    config: SimulationConfig,
    projected_claim: str = "popcorn",
) -> Tuple[Scenario, Mapping[str, Sequence[ProjectionTemplate]]]:
    """Build Sam's representation of the trusted friend at depth two.

    The base history says that Sam received popcorn testimony from the friend.
    The template determines what the observer represents Sam as taking the
    friend to believe. Matching and perturbed claims therefore produce
    different whole-LEWM coherence without changing the flagship scenarios.
    """

    scenario = build_variant("trusted", config)
    scenario.lewm.max_depth = 2
    belief_source = Source(
        "friend_belief_signal",
        projected_claim,
        "contents",
        "belief",
        0.0,
        1.0,
        1.0,
        [],
    )
    if belief_source.key not in scenario.lewm.state_nodes:
        scenario.lewm.state_nodes.append(belief_source.key)
    templates = {
        scenario.target: (
            ProjectionTemplate(
                "friend",
                (belief_source,),
                config.s1_default,
                1.0,
            ),
        )
    }
    scenario.lewm.validate()
    return scenario, templates
