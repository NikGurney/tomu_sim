"""Tests for worked-example and Run 2 scenario construction."""

from tomu.config import DEFAULT_CONFIG
from tomu.scenarios import build_all_variants, build_run2, build_variant


def test_variant_histories_match_worked_example() -> None:
    variants = {scenario.name: scenario for scenario in build_all_variants(DEFAULT_CONFIG)}
    trusted = variants["trusted"].lewm.agents["a2"].history
    assert [source.key for source in trusted] == ["testimony", "label"]
    assert variants["naive"].lewm.agents["a2"].history == []
    assert variants["motivated"].motivated_prior == ("a2", "o_pickle")
    assert "o_absent" not in variants["motivated"].lewm.state_nodes
    assert {"o_chocolate", "o_pickle", "o_popcorn"}.issubset(
        variants["motivated"].lewm.state_nodes
    )
    for scenario in variants.values():
        scenario.lewm.validate()


def test_scenarios_are_fresh() -> None:
    first = build_variant("trusted", DEFAULT_CONFIG)
    second = build_variant("trusted", DEFAULT_CONFIG)
    first.lewm.agents["a2"].history[0].cred = 0.2
    assert second.lewm.agents["a2"].history[0].cred == 1.0


def test_run2_has_three_targets_and_six_edges() -> None:
    lewm, targets, behaviors = build_run2(DEFAULT_CONFIG)
    assert targets == ["a2", "a3", "a4"]
    assert len(lewm.edges) == 6
    assert set(behaviors.values()) == {"CONTENT"}
    lewm.validate()
