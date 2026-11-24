"""Unit tests for the Definition 4 inference procedures and classifications."""

from dataclasses import replace

from tomu.config import DEFAULT_CONFIG
from tomu.agent_node import Source
from tomu.inference import (
    Mode,
    backward_inference,
    behavior_match,
    classify,
    generate_candidates,
    motivated_candidate_configs,
    mutual_reconciliation,
    self_projection,
    source_support,
)
from tomu.scenarios import build_variant


def test_candidate_generation_is_sorted_and_inert() -> None:
    scenario = build_variant("trusted", DEFAULT_CONFIG)
    history = scenario.lewm.agents[scenario.target].history
    assert generate_candidates(history, "contents") == ["chocolate", "popcorn"]
    assert generate_candidates(history, "contents", ["pickle"]) == ["pickle"]
    assert generate_candidates(history, "contents", subjective=["pickle"]) == [
        "chocolate", "pickle", "popcorn"
    ]


def test_behavior_gate() -> None:
    assert behavior_match("absent", "NEUTRAL") == 1.0
    assert behavior_match("popcorn", "NEUTRAL") == 0.0
    assert behavior_match("pickle", None) == 1.0


def test_discounted_typed_support_favors_testimony() -> None:
    scenario = build_variant("trusted", DEFAULT_CONFIG)
    sources = scenario.lewm.agents[scenario.target].history
    support = source_support("popcorn", sources, Mode(), DEFAULT_CONFIG)
    assert 0.93 < support < 0.95


def test_backward_inference_trusted_full_model() -> None:
    scenario = build_variant("trusted", DEFAULT_CONFIG)
    result = backward_inference(
        scenario.lewm.agents[scenario.target], scenario.behavior, "contents",
        Mode(), DEFAULT_CONFIG.s1_default, DEFAULT_CONFIG,
    )
    assert result.value == "popcorn"
    assert result.status == "ACCEPTED"


def test_shared_classification_boundaries() -> None:
    assert classify(DEFAULT_CONFIG.tau_high, DEFAULT_CONFIG) == "ACCEPTED_LOW"
    assert classify(DEFAULT_CONFIG.tau_low, DEFAULT_CONFIG) == "ACCEPTED_LOW"
    assert classify(DEFAULT_CONFIG.tau_low - 0.01, DEFAULT_CONFIG) == "REJECTED"


def test_pickle_configs_are_distinct_lewms_with_one_dependency() -> None:
    scenario = build_variant("motivated", DEFAULT_CONFIG)
    prior = scenario.motivated_prior
    assert prior is not None
    configs = motivated_candidate_configs(
        scenario.lewm.agents[scenario.target], scenario.behavior, prior, DEFAULT_CONFIG
    )
    assert [candidate.bls_scalar for candidate in configs] == [0.6, 0.8, 1.0]
    assert all(candidate.dependency_edges == frozenset({prior}) for candidate in configs)
    assert all(candidate.behavior_evidence == "CONTENT" for candidate in configs)


def test_mutual_reconciliation_reads_previous_round_selection() -> None:
    scenario = build_variant("trusted", DEFAULT_CONFIG)
    config = replace(DEFAULT_CONFIG, tau_high=0.99)
    outcome = mutual_reconciliation(
        scenario.lewm,
        [scenario.target],
        {scenario.target: scenario.behavior},
        "contents",
        2,
        "min",
        1.0,
        config,
    )
    assert len(outcome.rounds) == 2
    assert outcome.rounds[0].results["a2"].evaluated[0] == "chocolate"
    assert outcome.rounds[1].results["a2"].evaluated[0] == "popcorn"
    assert scenario.lewm.inference_records[("BI", "a2")].confidence == outcome.confidence
    assert ("a2", "o_popcorn") in scenario.lewm.edges


def test_self_projection_selects_own_content_from_filtered_focal_history() -> None:
    scenario = build_variant("trusted", DEFAULT_CONFIG)
    focal = scenario.lewm.agents[scenario.lewm.focal]
    focal.history = [
        Source(
            "observer_signal", "chocolate", "contents", "belief",
            0.0, 1.0, 1.0, [],
        )
    ]
    scenario.lewm.state_nodes.append("observer_signal")
    result = self_projection(
        scenario.lewm,
        scenario.target,
        "contents",
        DEFAULT_CONFIG.s1_default,
        DEFAULT_CONFIG,
        1.0,
    )
    assert result.value == "chocolate"
    assert result.value != "popcorn"
    assert scenario.lewm.agents["sp:a2/a1"].history[0].claim == "chocolate"
