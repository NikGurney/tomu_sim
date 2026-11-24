"""Agent-node component: ordered histories and projected perspectives."""

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Source:
    """One epistemic source in an agent's ordered history (Definition 2, manuscript §3.2)."""

    key: str
    claim: str
    bears_on: str
    kind: str
    trust: float
    obs: float
    cred: float
    discounts: List[str]
    speaker: Optional[str] = None


@dataclass
class AgentNode:
    """An agent, its ordered history, sophistication, and timestamp (Definition 2, manuscript §3.2)."""

    key: str
    history: List[Source]
    sophistication: int
    timestamp: float


@dataclass
class ProjectedAgentNode(AgentNode):
    """An agent as represented by another node in the proliferation tree."""

    represented_agent: str
    parent: str
    path: Tuple[str, ...]
