"""Bounded proliferation of concrete, target-relative agent projections."""

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .agent_node import ProjectedAgentNode, Source
from .config import SimulationConfig
from .lewm import Edge, LEWM


@dataclass(frozen=True)
class ProjectionTemplate:
    """Information from which a parent can construct one projected agent."""

    represented_agent: str
    history: Tuple[Source, ...]
    sophistication: int
    access: float = 1.0


@dataclass(frozen=True)
class Branch:
    """A tree reference to an actual agent node stored in the LEWM."""

    node: str
    represented_agent: str
    target: str
    parent: Optional[str]
    path: Tuple[str, ...]
    depth: int
    width: int
    running_confidence: float


def branching_width(k: int, s1: int, rho: float) -> int:
    """Compute ``b(k,S1)=floor(S1*rho**k)`` for projected depths."""

    return math.floor(s1 * rho**k)


# SIM CHOICE: illustrative form; the model does not fix this.
def marginal_confidence_gain(
    running_confidence: float, depth: int, config: SimulationConfig
) -> float:
    """Estimate a projected level's marginal gain for the KAPPA stop rule."""

    return running_confidence * config.rho**depth


def filter_history(history: Sequence[Source], access: float) -> List[Source]:
    """Project an ordered history through a target-relative access channel.

    Access composes multiplicatively with each source's observability. Sources
    with zero effective observability are absent from the projected history;
    retained sources preserve their original order.
    """

    projected: List[Source] = []
    for source in history:
        effective_obs = source.obs * access
        if effective_obs <= 0.0:
            continue
        projected.append(Source(
            source.key,
            source.claim,
            source.bears_on,
            source.kind,
            source.trust,
            effective_obs,
            source.cred,
            list(source.discounts),
            source.speaker,
        ))
    return projected


def projected_key(path: Sequence[str]) -> str:
    """Return a deterministic graph key for one projected path."""

    return "projection:" + "/".join(path)


def _drop_optional_projected_nodes(lewm: LEWM) -> None:
    """Remove rebuildable proliferation nodes while retaining SP projections."""

    removable = {
        key for key, agent in lewm.agents.items()
        if isinstance(agent, ProjectedAgentNode) and not key.startswith("sp:")
    }
    for key in removable:
        del lewm.agents[key]
    for key in list(lewm.edges):
        if key[0] in removable or key[1] in removable:
            del lewm.edges[key]
    for record_key, record in list(lewm.inference_records.items()):
        if record.subject in removable:
            del lewm.inference_records[record_key]


def _instantiate_child(
    lewm: LEWM,
    parent: Branch,
    template: ProjectionTemplate,
    access: float,
    timestamp: float,
) -> Branch:
    """Construct one projected agent and its parent modeling edge."""

    path = parent.path + (template.represented_agent,)
    key = projected_key(path)
    history = filter_history(template.history, access * template.access)
    sophistication = min(
        lewm.agents[parent.node].sophistication,
        template.sophistication,
    )
    lewm.agents[key] = ProjectedAgentNode(
        key,
        history,
        sophistication,
        timestamp,
        template.represented_agent,
        parent.node,
        path,
    )
    lewm.edges[(parent.node, key)] = Edge(
        parent.node,
        key,
        ("modeling", 1.0),
        access * template.access,
        1.0,
        report=False,
    )
    for source in history:
        lewm.edges[(key, source.key)] = Edge(
            key,
            source.key,
            (source.kind, source.trust),
            source.obs,
            source.cred,
            report=False,
        )
    prior = next(
        (
            record for record in reversed(lewm.inference_history)
            if record.procedure == "PROJECTION" and record.target == key
        ),
        None,
    )
    if prior is not None:
        lewm.inference_records[("PROJECTION", key)] = prior
        if prior.content_node is not None:
            lewm.edges[(key, prior.content_node)] = Edge(
                key,
                prior.content_node,
                ("belief", prior.bls_scalar),
                access * template.access,
                1.0,
                report=False,
            )
    return Branch(
        key,
        template.represented_agent,
        parent.target,
        parent.node,
        path,
        parent.depth + 1,
        1,
        parent.running_confidence,
    )


def rebuild_and_prune(
    lewm: LEWM,
    targets: Iterable[str],
    confidence_by_target: Dict[str, float],
    s1: int,
    config: SimulationConfig,
    projection_templates: Optional[
        Mapping[str, Sequence[ProjectionTemplate]]
    ] = None,
    timestamp: float = 0.0,
) -> List[Branch]:
    """Rebuild a bounded tree containing concrete projected agent nodes.

    Every non-focal target is mandatory at depth one. Branch width, confidence
    pruning, and the marginal-gain rule apply only below that target layer.
    Templates are keyed by the represented identity of the prospective parent.
    """

    _drop_optional_projected_nodes(lewm)
    templates = projection_templates or {}
    branches: List[Branch] = []
    frontier: List[Branch] = []
    for target in sorted(targets):
        if target not in lewm.agents:
            raise ValueError(f"target missing from LEWM: {target}")
        if (lewm.focal, target) not in lewm.edges:
            raise ValueError(f"modeling edge missing for target: {target}")
        branch = Branch(
            target,
            target,
            target,
            lewm.focal,
            (target,),
            1,
            1,
            confidence_by_target[target],
        )
        branches.append(branch)
        frontier.append(branch)

    while frontier:
        parent = frontier.pop(0)
        depth = parent.depth + 1
        target_sophistication = lewm.agents[parent.target].sophistication
        effective_ceiling = min(
            s1,
            target_sophistication,
            lewm.max_depth,
        )
        if depth > effective_ceiling:
            continue
        width = branching_width(depth, s1, config.rho)
        if width <= 0 or parent.running_confidence < config.tau_low:
            continue
        if marginal_confidence_gain(
            parent.running_confidence, depth, config
        ) < config.kappa:
            continue
        candidates = list(templates.get(parent.represented_agent, ()))[:width]
        for template in candidates:
            parent_edge = (parent.parent, parent.node)
            access = lewm.edges[parent_edge].obs if parent_edge in lewm.edges else 1.0
            child = _instantiate_child(
                lewm, parent, template, access, timestamp
            )
            child = Branch(
                child.node,
                child.represented_agent,
                child.target,
                child.parent,
                child.path,
                child.depth,
                width,
                child.running_confidence,
            )
            branches.append(child)
            frontier.append(child)

    lewm.validate()
    return branches
