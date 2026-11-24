"""LEWM graph, typed edges, and off-graph inference records."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .agent_node import AgentNode

EdgeKey = Tuple[str, str]
RecordKey = Tuple[str, str]


@dataclass
class Edge:
    """A directed relationship whose BLS value is ``(type, scalar)``.

    Belief content is represented by ``dst`` when the edge targets a state
    node. Inference confidence is deliberately not stored here.
    """

    src: str
    dst: str
    belief: Optional[Tuple[str, float]]
    obs: float
    cred: float
    floor: float = 0.0
    acute: float = 0.0
    t_last: float = 0.0
    report: bool = True


@dataclass(frozen=True)
class InferenceRecord:
    """One operative selection, kept outside the LEWM edge relation."""

    procedure: str
    target: str
    subject: str
    content_node: Optional[str]
    bls_scalar: float
    confidence: float
    timestamp: float


@dataclass
class LEWM:
    """The local epistemic world model W from Definition 1."""

    focal: str
    agents: Dict[str, AgentNode]
    state_nodes: List[str]
    edges: Dict[EdgeKey, Edge]
    inference_records: Dict[RecordKey, InferenceRecord] = field(default_factory=dict)
    inference_history: List[InferenceRecord] = field(default_factory=list)
    max_depth: int = 1

    def validate(self) -> None:
        """Require disjoint node sets and graph-closed edge endpoints."""

        agent_keys = set(self.agents)
        state_keys = set(self.state_nodes)
        if agent_keys & state_keys:
            raise ValueError("agent and state node sets must be disjoint")
        if self.focal not in agent_keys:
            raise ValueError("focal agent must belong to A")
        nodes = agent_keys | state_keys
        for key, edge in self.edges.items():
            if key != (edge.src, edge.dst):
                raise ValueError(f"edge key does not match endpoints: {key}")
            if edge.src not in nodes or edge.dst not in nodes:
                raise ValueError(f"edge endpoint outside A union O: {key}")
            if not 0.0 <= edge.obs <= 1.0:
                raise ValueError(f"edge observability outside [0,1]: {key}")
            if not 0.0 < edge.cred <= 1.0:
                raise ValueError(f"edge credibility outside (0,1]: {key}")
            if edge.belief is not None and not 0.0 <= edge.belief[1] <= 1.0:
                raise ValueError(f"BLS scalar outside [0,1]: {key}")
        for record in self.inference_records.values():
            if record.subject not in agent_keys:
                raise ValueError(
                    f"inference-record subject outside agent set: {record.subject}"
                )
            if (
                record.content_node is not None
                and record.content_node not in state_keys
            ):
                raise ValueError(
                    "inference-record content outside state-node set: "
                    f"{record.content_node}"
                )
