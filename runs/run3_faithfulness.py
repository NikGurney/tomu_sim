"""Run 3: four worked-example variants as a faithfulness check."""

import random
from typing import Dict, List, Tuple

from tomu.config import DEFAULT_CONFIG, SimulationConfig
from tomu.inference import Mode, Result, backward_inference, run_motivated_episode
from tomu.scenarios import Scenario, build_all_variants
from tomu.trace import InferenceTrace

from .common import aligned_table, write_csv, write_text

TITLE = (
    "Faithfulness check — verifies the implementation matches the worked example. "
    "Not evidence for the theory."
)


def _ordinary_row(scenario: Scenario, result: Result) -> Dict[str, object]:
    return {
        "variant": scenario.name,
        "attribution": result.value,
        "confidence": result.confidence,
        "classification": result.status,
        "ell": result.ell,
        "implicated_floored": False,
        "session1_attribution": "",
        "session1_confidence": "",
        "session1_classification": "",
        "session1_ell": "",
        "residue_edge": "",
        "post_rejection_cred": "",
        "session2_attribution": "",
        "session2_confidence": "",
        "session2_classification": "",
        "session2_ell": "",
    }


def _motivated_row(
    scenario: Scenario,
    config: SimulationConfig,
    traces: Dict[str, InferenceTrace],
) -> Dict[str, object]:
    episode_traces: List[InferenceTrace] = []
    episode = run_motivated_episode(scenario, config, traces=episode_traces)
    for index, trace in enumerate(episode_traces, start=1):
        traces[f"motivated/session{index}"] = trace
    first, second = episode.sessions
    residue_edge = episode.residue_edge
    return {
        "variant": "motivated",
        "attribution": episode.attribution,
        "confidence": episode.confidence,
        "classification": episode.classification,
        "ell": second.result.ell,
        "implicated_floored": scenario.lewm.edges[residue_edge].floor > 0.0,
        "session1_attribution": first.result.value,
        "session1_confidence": first.result.confidence,
        "session1_classification": first.result.status,
        "session1_ell": first.result.ell,
        "residue_edge": f"{residue_edge[0]}->{residue_edge[1]}",
        "post_rejection_cred": episode.post_rejection_credibility,
        "session2_attribution": second.result.value,
        "session2_confidence": second.result.confidence,
        "session2_classification": second.result.status,
        "session2_ell": second.result.ell,
    }


def evaluate(
    config: SimulationConfig = DEFAULT_CONFIG,
) -> Tuple[List[Dict[str, object]], Dict[str, InferenceTrace]]:
    """Infer all four worked-example variants with an explicit episode record."""

    rows: List[Dict[str, object]] = []
    traces: Dict[str, InferenceTrace] = {}
    for scenario in build_all_variants(config):
        if scenario.name == "motivated":
            rows.append(_motivated_row(scenario, config, traces))
            continue
        trace = InferenceTrace()
        target = scenario.lewm.agents[scenario.target]
        result = backward_inference(
            target,
            scenario.behavior,
            "contents",
            Mode(),
            config.s1_default,
            config,
            trace=trace,
            focal=scenario.lewm.focal,
        )
        rows.append(_ordinary_row(scenario, result))
        traces[scenario.name] = trace
    return rows, traces


def assert_acceptance(rows: List[Dict[str, object]]) -> None:
    """Assert the four worked-example outcomes and separated episode state."""

    by_name = {row["variant"]: row for row in rows}
    assert by_name["standard"]["attribution"] == "chocolate"
    assert by_name["standard"]["ell"] == 1
    assert by_name["standard"]["classification"] == "ACCEPTED"
    assert by_name["trusted"]["attribution"] == "popcorn"
    assert by_name["trusted"]["classification"] == "ACCEPTED"
    assert by_name["naive"]["attribution"] == "absent"
    motivated = by_name["motivated"]
    assert motivated["attribution"] == "popcorn"
    assert motivated["classification"] == "ACCEPTED"
    assert motivated["session1_classification"] == "REJECTED"
    assert motivated["session1_ell"] == 3
    assert motivated["implicated_floored"] is True
    assert motivated["residue_edge"] == "a2->o_pickle"


def render(rows: List[Dict[str, object]], traces: Dict[str, InferenceTrace]) -> str:
    """Render terminal outcomes and unmerged per-session traces."""

    display = [
        [
            row["variant"], row["attribution"], f"{float(row['confidence']):.6f}",
            row["classification"], row["ell"], row["implicated_floored"],
        ]
        for row in rows
    ]
    table = aligned_table(
        ["variant", "attribution", "confidence", "classification", "ell", "floored"],
        display,
    )
    motivated = next(row for row in rows if row["variant"] == "motivated")
    episode_table = aligned_table(
        ["session", "attribution", "confidence", "classification", "ell"],
        [
            [1, motivated["session1_attribution"],
             f"{float(motivated['session1_confidence']):.6f}",
             motivated["session1_classification"], motivated["session1_ell"]],
            [2, motivated["session2_attribution"],
             f"{float(motivated['session2_confidence']):.6f}",
             motivated["session2_classification"], motivated["session2_ell"]],
        ],
    )
    episode_note = (
        "Motivated episode\n"
        + episode_table
        + f"\nresidue edge: {motivated['residue_edge']}"
        + f"\npost-rejection cred: {float(motivated['post_rejection_cred']):.6f}"
    )
    trace_text = []
    for name in [
        "standard", "trusted", "naive", "motivated/session1", "motivated/session2"
    ]:
        trace_text.append(f"Trace: {name}\n{traces[name].transcript()}")
    return "\n\n".join([TITLE, table, episode_note] + trace_text)


def main() -> None:
    """Execute Run 3 and write explicit terminal/session artifacts."""

    random.seed(DEFAULT_CONFIG.seed)
    rows, traces = evaluate()
    assert_acceptance(rows)
    text = render(rows, traces)
    write_text("run3_faithfulness.txt", text)
    numeric_fields = {
        "confidence", "session1_confidence", "post_rejection_cred",
        "session2_confidence",
    }
    csv_rows = []
    for row in rows:
        csv_row = dict(row)
        for field in numeric_fields:
            if csv_row[field] != "":
                csv_row[field] = f"{float(csv_row[field]):.12f}"
        csv_rows.append(csv_row)
    fields = [
        "variant", "attribution", "confidence", "classification", "ell",
        "implicated_floored", "session1_attribution", "session1_confidence",
        "session1_classification", "session1_ell", "residue_edge",
        "post_rejection_cred", "session2_attribution", "session2_confidence",
        "session2_classification", "session2_ell",
    ]
    write_csv("run3_faithfulness.csv", csv_rows, fields)
    print(text)


if __name__ == "__main__":
    main()
