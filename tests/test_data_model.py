"""Tests for the Definition 1-2 data structures and central configuration."""

from tomu.agent_node import AgentNode, Source
from tomu.config import DEFAULT_CONFIG
from tomu.lewm import Edge, LEWM


def test_data_model_holds_typed_world() -> None:
    source = Source("s", "popcorn", "contents", "trust", 0.9, 0.8, 1.0, [])
    agent = AgentNode("a2", [source], 3, 0.0)
    focal = AgentNode("a1", [], 3, 0.0)
    edge = Edge("a2", "o_popcorn", ("belief", 1.0), 0.8, 1.0)
    world = LEWM(
        "a1", {"a1": focal, "a2": agent}, ["contents", "o_popcorn"],
        {("a2", "o_popcorn"): edge},
    )
    world.validate()
    assert world.agents["a2"].history[0].kind == "trust"
    assert world.edges[("a2", "o_popcorn")].belief == ("belief", 1.0)
    assert DEFAULT_CONFIG.s_max == 6
