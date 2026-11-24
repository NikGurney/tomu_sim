"""Theory of Mind Utility (ToM-U) reference simulation.

Notation map (manuscript §3, Definitions 1-5):

=======================  =================================  ================================
Model quantity           Code                               Meaning
=======================  =================================  ================================
LEWM W                   ``LEWM``                           world model
agent/state sets         ``lewm.agents/state_nodes``       node collections
edge e                   ``Edge``                           directed typed relationship
BLS(e)                   ``edge.belief``                    (edge type, BLS scalar)
obs(e)                   ``edge.obs``                       observability in [0, 1]
cred(e)                  ``edge.cred``                      residue-derived credibility
agent a_i                ``AgentNode``                      one agent
H_i                      ``agent.history``                  ordered exposure history
S_i, S_max               ``sophistication``, ``S_MAX``      sophistication and ceiling
b(k, S_1)                ``branching_width(k, s1)``         width at depth k
beta_i                   ``behavior``                       observed behavior
f_j                      ``fit``                            candidate fit in [0, 1]
C_l                      ``confidence``                     accumulated confidence
selected inference       ``lewm.inference_records``         off-graph content/confidence
R(e, t)                  ``residue(edge, t)``               edge residue at time t
F(e)                     ``edge.floor``                     persistent residue floor
=======================  =================================  ================================
"""

from .agent_node import AgentNode, ProjectedAgentNode, Source
from .lewm import Edge, InferenceRecord, LEWM

__all__ = [
    "AgentNode", "Edge", "InferenceRecord", "LEWM", "ProjectedAgentNode", "Source"
]
