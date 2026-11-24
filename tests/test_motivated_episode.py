"""Causal and regime checks for the motivated commitment gate."""

from dataclasses import replace

from tomu.config import DEFAULT_CONFIG
from tomu.inference import run_motivated_episode, run_motivated_sessions
from tomu.scenarios import build_variant


def test_residue_causes_next_session_fallback() -> None:
    with_residue = build_variant("motivated", DEFAULT_CONFIG)
    adaptive = run_motivated_episode(with_residue, DEFAULT_CONFIG)
    assert adaptive.sessions[0].committed is True
    assert adaptive.sessions[0].result.status == "REJECTED"
    assert adaptive.sessions[0].result.ell == DEFAULT_CONFIG.s1_default
    assert adaptive.sessions[1].committed is False
    assert adaptive.fallback_reached is True
    assert adaptive.attribution == "popcorn"
    assert adaptive.classification == "ACCEPTED"

    without_residue = build_variant("motivated", DEFAULT_CONFIG)
    counterfactual = run_motivated_episode(
        without_residue, DEFAULT_CONFIG, apply_residue_update=False
    )
    assert all(session.committed for session in counterfactual.sessions)
    assert all(session.result.status == "REJECTED" for session in counterfactual.sessions)
    assert counterfactual.fallback_reached is False


def test_only_pickle_prior_receives_residue() -> None:
    scenario = build_variant("motivated", DEFAULT_CONFIG)
    episode = run_motivated_episode(scenario, DEFAULT_CONFIG)
    assert scenario.lewm.edges[episode.residue_edge].floor > 0.0
    assert scenario.lewm.edges[("a2", "label")].floor == 0.0
    assert scenario.lewm.edges[("a2", "testimony")].floor == 0.0


def test_pathological_zero_threshold_preserves_ten_session_commitment() -> None:
    config = replace(DEFAULT_CONFIG, tau_commit=0.0)
    scenario = build_variant("motivated", config)
    episode = run_motivated_sessions(scenario, config, session_count=10)
    assert len(episode.sessions) == 10
    assert all(session.committed for session in episode.sessions)
    assert all(session.result.status == "REJECTED" for session in episode.sessions)
    assert episode.fallback_reached is False
