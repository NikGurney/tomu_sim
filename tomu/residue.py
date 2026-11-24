"""Residue component: decay, persistence, rehabilitation, and targeting (Sections 8.3/8.6)."""

import math
from typing import Iterable, List, Set

from .agent_node import AgentNode, Source
from .config import SimulationConfig
from .lewm import Edge, EdgeKey, LEWM


def residue(edge: Edge, t: float, config: SimulationConfig) -> float:
    """Return R(e,t), the decayed acute residue plus floor (Appendix A.3)."""

    elapsed = max(0.0, t - edge.t_last)
    return edge.acute * math.exp(-config.decay_rate * elapsed) + edge.floor


# SIM CHOICE: illustrative form; the model does not fix this.
def credibility(edge: Edge, t: float, config: SimulationConfig) -> float:
    """Map R(e,t) monotonically to positive credibility (Appendix A.3)."""

    return 1.0 / (1.0 + residue(edge, t, config))


def recompute_cred(edge: Edge, t: float, config: SimulationConfig) -> None:
    """Synchronize stored edge credibility with its residue at t (Appendix A.3)."""

    edge.cred = credibility(edge, t, config)


def accrue_residue(
    edge: Edge, disconfirmation: float, t: float, config: SimulationConfig
) -> None:
    """Mutate an implicated edge after rejection using degree d (Appendix A.3, Eq. A1)."""

    current_acute = residue(edge, t, config) - edge.floor
    edge.acute = current_acute + config.alpha * disconfirmation
    edge.floor = min(1.0, edge.floor + config.beta * disconfirmation)
    edge.t_last = t
    recompute_cred(edge, t, config)


def rehabilitate(
    edge: Edge, signal: float, t: float, config: SimulationConfig
) -> None:
    """Partly reduce acute residue without lowering its floor (Appendix A.3, Eq. A2)."""

    current_acute = residue(edge, t, config) - edge.floor
    edge.acute = max(0.0, current_acute - config.gamma * signal)
    edge.t_last = t
    recompute_cred(edge, t, config)


def implicated_edges(
    focal: str, target: AgentNode, sources: Iterable[Source], o_star: str
) -> Set[EdgeKey]:
    """Return estimate and relied-on source edges for an account (Definition 5; Appendix A.2)."""

    del o_star  # retained in the signature to mirror the model notation
    edges = {(focal, target.key)}
    edges.update((target.key, source.key) for source in sources)
    return edges


def apply_to_edges(
    lewm: LEWM,
    edge_keys: Iterable[EdgeKey],
    disconfirmation: float,
    t: float,
    config: SimulationConfig,
) -> None:
    """Accrue residue on sorted implicated keys and nowhere else (Definition 5, targeting condition)."""

    for edge_key in sorted(edge_keys):
        accrue_residue(lewm.edges[edge_key], disconfirmation, t, config)


def sync_source_credibilities(lewm: LEWM) -> None:
    """Copy source-edge credibility into matching Source records before inference."""

    for agent_key in sorted(lewm.agents):
        agent = lewm.agents[agent_key]
        for source in agent.history:
            edge_key = (agent_key, source.key)
            if edge_key in lewm.edges:
                source.cred = lewm.edges[edge_key].cred


def sync_credibilities(lewm: LEWM, t: float, config: SimulationConfig) -> None:
    """Refresh every decay-aware edge credibility and its history mirror."""

    for edge in lewm.edges.values():
        recompute_cred(edge, t, config)
    sync_source_credibilities(lewm)


def floored_edges(lewm: LEWM) -> List[EdgeKey]:
    """Return deterministic keys of edges whose persistent floor is positive."""

    return [key for key in sorted(lewm.edges) if lewm.edges[key].floor > 0.0]
