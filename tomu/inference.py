"""Candidate inference and stateful mutual reconciliation for ToM-U."""

from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING, Callable, Dict, FrozenSet, Iterable, List, Mapping,
    Optional, Sequence, Set, Tuple,
)

from .agent_node import AgentNode, ProjectedAgentNode, Source
from .config import CONTENT_NODE_BY_VALUE, SimulationConfig
from .lewm import Edge, EdgeKey, InferenceRecord, LEWM
from .proliferation import (
    Branch, ProjectionTemplate, filter_history, rebuild_and_prune,
)
from .residue import (
    apply_to_edges,
    credibility,
    recompute_cred,
    sync_credibilities,
    sync_source_credibilities,
)
from .trace import InferenceTrace

if TYPE_CHECKING:
    from .scenarios import Scenario

VALUE_BY_CONTENT_NODE = {
    node: value for value, node in CONTENT_NODE_BY_VALUE.items()
}


@dataclass(frozen=True)
class Mode:
    """Ablate source-discount application and source typing."""

    apply_discounts: bool = True
    typed: bool = True


@dataclass(frozen=True)
class CandidateConfig:
    """One discrete candidate LEWM configuration."""

    value: str
    content_node: Optional[str]
    bls_scalar: float
    dependency_edges: FrozenSet[EdgeKey]
    evaluation_edges: FrozenSet[EdgeKey]
    behavior_evidence: Optional[str]
    label: str


@dataclass(frozen=True)
class Result:
    """Terminal backward-inference result and its residue dependencies."""

    value: str
    confidence: float
    best_fit: float
    implicated: Set[EdgeKey]
    ell: int
    status: str
    content_node: Optional[str] = None
    bls_scalar: float = 1.0
    evaluated: Tuple[str, ...] = ()


@dataclass(frozen=True)
class RoundRecord:
    """Observable record that one MR round consumed its current tree/state."""

    index: int
    timestamp: float
    branches: Tuple[Branch, ...]
    results: Dict[str, Result]
    pair: Dict[str, float]
    projections: Dict[str, float]
    projection_results: Dict[str, Result] = field(default_factory=dict)
    branch_results: Dict[str, Result] = field(default_factory=dict)
    branch_coherence: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Outcome:
    """Terminal mutual-reconciliation status and per-round evidence."""

    status: str
    confidence: float
    pair: Dict[str, float]
    results: Dict[str, Result]
    rounds: Tuple[RoundRecord, ...] = field(default_factory=tuple)
    projection_results: Dict[str, Result] = field(default_factory=dict)
    terminal_timestamp: float = 0.0


@dataclass(frozen=True)
class SessionRecord:
    """One invocation in a motivated multi-session episode."""

    index: int
    timestamp: float
    committed: bool
    prior_credibility: float
    result: Result


@dataclass(frozen=True)
class EpisodeRecord:
    """Unmerged motivated-session results and the terminal episode outcome."""

    sessions: Tuple[SessionRecord, ...]
    residue_edge: EdgeKey
    post_rejection_credibility: float
    fallback_reached: bool
    attribution: str
    confidence: float
    classification: str


def compatible(value: str) -> Set[str]:
    """Return behaviors compatible with a candidate content attribution."""

    if value == "absent":
        return {"NEUTRAL"}
    return {"CONTENT"}


def behavior_match(value: str, behavior: Optional[str]) -> float:
    """Return the behavior gate; behavior does not choose concrete content."""

    if behavior is None:
        return 1.0
    return 1.0 if behavior in compatible(value) else 0.0


# SIM CHOICE: illustrative form; the model does not fix this.
def generate_candidates(
    target_history: Iterable[Source],
    o_star: str,
    forced: Optional[Iterable[str]] = None,
    subjective: Optional[Iterable[str]] = None,
) -> List[str]:
    """Generate deterministic candidate values, including subjective relevance."""

    if forced is not None:
        return list(forced)
    sources = [source for source in target_history if source.bears_on == o_star]
    claims = {source.claim for source in sources}
    if subjective is not None:
        claims.update(subjective)
    return sorted(claims) if claims else ["absent"]


def _discounted_keys(sources: Iterable[Source], mode: Mode) -> Set[str]:
    """Collect sources explicitly discounted by other sources."""

    discounted: Set[str] = set()
    if mode.apply_discounts:
        for source in sources:
            discounted.update(source.discounts)
    return discounted


def _source_weight(
    source: Source,
    discounted: Set[str],
    mode: Mode,
    config: SimulationConfig,
) -> float:
    """Return one credibility/observability-weighted source vote."""

    if not mode.typed:
        strength = config.untyped_strength
    elif source.kind == "trust":
        strength = source.trust
    else:
        strength = config.belief_source_strength
    weight = strength * source.obs * source.cred
    if source.key in discounted:
        weight *= config.discount_factor
    return weight


def source_support(
    value: str, sources: List[Source], mode: Mode, config: SimulationConfig
) -> float:
    """Compute the credibility-weighted source vote share for ``value``."""

    discounted = _discounted_keys(sources, mode)
    weights = {
        source.key: _source_weight(source, discounted, mode, config)
        for source in sources
    }
    total = sum(weights.values())
    if total == 0.0:
        return 1.0 if value == "absent" else 0.0
    support = sum(weights[source.key] for source in sources if source.claim == value)
    return support / total


# SIM CHOICE: illustrative form; the model does not fix this.
def fit(
    value: str,
    behavior: Optional[str],
    sources: List[Source],
    mode: Mode,
    config: SimulationConfig,
) -> float:
    """Multiply behavior compatibility by information-history support."""

    return behavior_match(value, behavior) * source_support(value, sources, mode, config)


def classify(confidence: float, config: SimulationConfig) -> str:
    """Apply the shared three-state confidence classification."""

    if confidence > config.tau_high:
        return "ACCEPTED"
    if confidence >= config.tau_low:
        return "ACCEPTED_LOW"
    return "REJECTED"


def _halt_reason(
    confidence: float, ell: int, candidate_count: int, s1: int, config: SimulationConfig
) -> str:
    if confidence > config.tau_high:
        return "confidence exceeded TAU_HIGH"
    if ell >= s1:
        return "sophistication ceiling reached"
    if ell >= candidate_count:
        return "candidates exhausted"
    return "continued"


def _normal_configs(
    target: AgentNode,
    sources: List[Source],
    behavior: Optional[str],
    o_star: str,
    focal: str,
    subjective: Optional[Iterable[str]],
    subjective_dependencies: Optional[Dict[str, EdgeKey]] = None,
) -> List[CandidateConfig]:
    values = generate_candidates(target.history, o_star, subjective=subjective)
    evaluation = frozenset((target.key, source.key) for source in sources)
    configs: List[CandidateConfig] = []
    for value in values:
        relied_on = {
            (target.key, source.key) for source in sources if source.claim == value
        }
        if subjective_dependencies is not None and value in subjective_dependencies:
            dependencies = frozenset({subjective_dependencies[value]})
        else:
            dependencies = frozenset({(focal, target.key)} | relied_on)
        configs.append(CandidateConfig(
            value=value,
            content_node=CONTENT_NODE_BY_VALUE.get(value),
            bls_scalar=1.0,
            dependency_edges=dependencies,
            evaluation_edges=evaluation,
            behavior_evidence=behavior,
            label=value,
        ))
    return configs


def _prior_value_first(
    candidates: List[CandidateConfig], prior_value: Optional[str]
) -> List[CandidateConfig]:
    """Make the previous selected content the first viable next-round candidate."""

    if prior_value is None or not any(c.value == prior_value for c in candidates):
        return candidates
    return sorted(candidates, key=lambda c: (c.value != prior_value, c.label))


def _evaluate_candidate_configs(
    candidates: Sequence[CandidateConfig],
    score: Callable[[CandidateConfig], float],
    s1: int,
    config: SimulationConfig,
    trace: Optional[InferenceTrace],
) -> Result:
    """Evaluate a finite candidate set with the shared confidence accumulator.

    Candidate exhaustion is an explicit algorithmic-level stopping condition:
    exhaustive finite enumeration may finish before the sophistication ceiling.
    """

    if not candidates:
        raise ValueError("inference requires at least one candidate")
    confidence = 0.0
    best_candidate = candidates[0]
    best_fit = -1.0
    evaluated: List[str] = []
    ell = 0
    for ell, candidate in enumerate(candidates, start=1):
        candidate_fit = score(candidate)
        confidence = 1.0 - (1.0 - confidence) * (1.0 - candidate_fit)
        if candidate_fit > best_fit:
            best_candidate, best_fit = candidate, candidate_fit
        evaluated.append(candidate.label)
        if trace is not None:
            trace.log(ell, candidate.label, candidate_fit, confidence)
        if confidence > config.tau_high or ell >= s1:
            break
    if trace is not None:
        trace.halt(_halt_reason(confidence, ell, len(candidates), s1, config))
    return Result(
        value=best_candidate.value,
        confidence=confidence,
        best_fit=best_fit,
        implicated=set(best_candidate.dependency_edges),
        ell=ell,
        status=classify(confidence, config),
        content_node=best_candidate.content_node,
        bls_scalar=best_candidate.bls_scalar,
        evaluated=tuple(evaluated),
    )


def backward_inference(
    target: AgentNode,
    behavior: Optional[str],
    o_star: str,
    mode: Mode,
    s1: int,
    config: SimulationConfig,
    forced: Optional[Iterable[str]] = None,
    trace: Optional[InferenceTrace] = None,
    focal: str = "a1",
    candidate_configs: Optional[Iterable[CandidateConfig]] = None,
    subjective: Optional[Iterable[str]] = None,
    subjective_dependencies: Optional[Dict[str, EdgeKey]] = None,
    prior_value: Optional[str] = None,
) -> Result:
    """Infer a target belief by evaluating discrete candidate LEWMs."""

    sources = [source for source in target.history if source.bears_on == o_star]
    if candidate_configs is not None:
        candidates = list(candidate_configs)
    elif forced is not None:
        values = list(forced)
        candidates = _normal_configs(target, sources, behavior, o_star, focal, values)
        candidates = [candidate for candidate in candidates if candidate.value in values]
    else:
        candidates = _normal_configs(
            target,
            sources,
            behavior,
            o_star,
            focal,
            subjective,
            subjective_dependencies,
        )
    candidates = _prior_value_first(candidates, prior_value)
    return _evaluate_candidate_configs(
        candidates,
        lambda candidate: fit(
            candidate.value, behavior, sources, mode, config
        ),
        s1,
        config,
        trace,
    )


def motivated_candidate_configs(
    target: AgentNode,
    behavior: Optional[str],
    prior_edge: EdgeKey,
    config: SimulationConfig,
) -> List[CandidateConfig]:
    """Return the three deterministic pickle LEWMs for committed search."""

    evaluation = frozenset((target.key, source.key) for source in target.history)
    return [
        CandidateConfig(
            value="pickle",
            content_node=CONTENT_NODE_BY_VALUE["pickle"],
            bls_scalar=scalar,
            dependency_edges=frozenset({prior_edge}),
            evaluation_edges=evaluation,
            behavior_evidence=behavior,
            label=f"pickle[{scalar:.1f}]",
        )
        for scalar in config.pickle_bls_scalars
    ]


def _write_back_selected_belief(
    lewm: LEWM,
    procedure: str,
    target: str,
    subject: str,
    result: Result,
    timestamp: float,
    observability: float,
) -> None:
    """Write content to the graph and selection/confidence to a record."""

    previous = lewm.inference_records.get((procedure, target))
    if (
        previous is not None
        and previous.content_node is not None
        and previous.content_node != result.content_node
    ):
        previous_key = (previous.subject, previous.content_node)
        previous_edge = lewm.edges.get(previous_key)
        if (
            previous_edge is not None
            and not previous_edge.report
            and previous_edge.floor == 0.0
            and previous_edge.acute == 0.0
        ):
            del lewm.edges[previous_key]
    if result.content_node is not None:
        edge_key = (subject, result.content_node)
        existing = lewm.edges.get(edge_key)
        cred = existing.cred if existing is not None else 1.0
        floor = existing.floor if existing is not None else 0.0
        acute = existing.acute if existing is not None else 0.0
        t_last = existing.t_last if existing is not None else 0.0
        lewm.edges[edge_key] = Edge(
            subject,
            result.content_node,
            ("belief", result.bls_scalar),
            observability,
            cred,
            floor,
            acute,
            t_last,
            report=False,
        )
    record = InferenceRecord(
        procedure,
        target,
        subject,
        result.content_node,
        result.bls_scalar,
        result.confidence,
        timestamp,
    )
    lewm.inference_records[(procedure, target)] = record
    lewm.inference_history.append(record)
    lewm.agents[subject].timestamp = timestamp


def run_motivated_sessions(
    scenario: "Scenario",
    config: SimulationConfig,
    session_count: int,
    apply_residue_update: bool = True,
    traces: Optional[List[InferenceTrace]] = None,
) -> EpisodeRecord:
    """Run a decay-aware commitment episode for a fixed number of sessions."""

    if scenario.motivated_prior is None:
        raise ValueError("motivated scenario requires an explicit prior edge")
    prior_edge = scenario.motivated_prior
    target = scenario.lewm.agents[scenario.target]
    session_records: List[SessionRecord] = []
    post_rejection_credibility = scenario.lewm.edges[prior_edge].cred
    fallback_reached = False

    for index in range(1, session_count + 1):
        timestamp = 1.0 + (index - 1) * config.delta_t_session
        edge = scenario.lewm.edges[prior_edge]
        current_credibility = credibility(edge, timestamp, config)
        recompute_cred(edge, timestamp, config)
        committed = current_credibility >= config.tau_commit
        trace = InferenceTrace()
        if traces is not None:
            traces.append(trace)
        if committed:
            candidates = motivated_candidate_configs(
                target, scenario.behavior, prior_edge, config
            )
            result = backward_inference(
                target,
                scenario.behavior,
                "contents",
                Mode(),
                config.s1_default,
                config,
                candidate_configs=candidates,
                trace=trace,
                focal=scenario.lewm.focal,
            )
        else:
            result = backward_inference(
                target,
                scenario.behavior,
                "contents",
                Mode(),
                config.s1_default,
                config,
                subjective=["pickle"],
                subjective_dependencies={"pickle": prior_edge},
                trace=trace,
                focal=scenario.lewm.focal,
            )
            fallback_reached = result.value == "popcorn" and result.status != "REJECTED"

        session_records.append(SessionRecord(
            index, timestamp, committed, current_credibility, result
        ))
        if result.status == "REJECTED" and apply_residue_update:
            apply_to_edges(
                scenario.lewm,
                result.implicated,
                1.0 - result.best_fit,
                timestamp,
                config,
            )
            post_rejection_credibility = scenario.lewm.edges[prior_edge].cred
            sync_source_credibilities(scenario.lewm)
        elif result.status != "REJECTED":
            _write_back_selected_belief(
                scenario.lewm,
                "BI",
                scenario.target,
                scenario.target,
                result,
                timestamp,
                scenario.lewm.edges[(scenario.lewm.focal, scenario.target)].obs,
            )
        scenario.lewm.validate()

    terminal = session_records[-1].result
    return EpisodeRecord(
        sessions=tuple(session_records),
        residue_edge=prior_edge,
        post_rejection_credibility=post_rejection_credibility,
        fallback_reached=fallback_reached,
        attribution=terminal.value,
        confidence=terminal.confidence,
        classification=terminal.status,
    )


def run_motivated_episode(
    scenario: "Scenario",
    config: SimulationConfig,
    apply_residue_update: bool = True,
    traces: Optional[List[InferenceTrace]] = None,
) -> EpisodeRecord:
    """Run the two-session flagship motivated episode."""

    return run_motivated_sessions(
        scenario, config, 2, apply_residue_update=apply_residue_update, traces=traces
    )


def _sp_key(focal: str, target: str) -> str:
    """Return the stable key for the focal agent as projected by a target."""

    return f"sp:{target}/{focal}"


def _ensure_self_projection_node(
    lewm: LEWM, target: str, timestamp: float
) -> ProjectedAgentNode:
    """Construct the focal agent as it appears from one target's perspective."""

    visibility = lewm.edges[(lewm.focal, target)].obs
    key = _sp_key(lewm.focal, target)
    node = ProjectedAgentNode(
        key,
        filter_history(lewm.agents[lewm.focal].history, visibility),
        min(
            lewm.agents[lewm.focal].sophistication,
            lewm.agents[target].sophistication,
        ),
        timestamp,
        lewm.focal,
        target,
        (target, lewm.focal),
    )
    lewm.agents[key] = node
    lewm.edges[(target, key)] = Edge(
        target,
        key,
        ("modeling", 1.0),
        visibility,
        1.0,
        report=False,
    )
    for source in node.history:
        lewm.edges[(key, source.key)] = Edge(
            key,
            source.key,
            (source.kind, source.trust),
            source.obs,
            source.cred,
            report=False,
        )
    return node


# SIM CHOICE: target-relative observability scales candidate fit.
def self_projection(
    lewm: LEWM,
    target: str,
    o_star: str,
    s1: int,
    config: SimulationConfig,
    timestamp: float,
    trace: Optional[InferenceTrace] = None,
    prior_value: Optional[str] = None,
) -> Result:
    """Infer how the focal agent's beliefs appear to one target agent.

    This is a distinct generate-and-filter pass over a target-filtered copy of
    ``H_1``. It never borrows the content selected by backward inference.
    """

    projected = _ensure_self_projection_node(lewm, target, timestamp)
    sources = [source for source in projected.history if source.bears_on == o_star]
    candidates = _normal_configs(
        projected,
        sources,
        None,
        o_star,
        target,
        None,
    )
    candidates = _prior_value_first(candidates, prior_value)
    visibility = lewm.edges[(lewm.focal, target)].obs
    return _evaluate_candidate_configs(
        candidates,
        lambda candidate: visibility * source_support(
            candidate.value, sources, Mode(), config
        ),
        s1,
        config,
        trace,
    )


def _record_value(lewm: LEWM, procedure: str, target: str) -> Optional[str]:
    record = lewm.inference_records.get((procedure, target))
    if record is None or record.content_node is None:
        return None
    edge = lewm.edges.get((record.subject, record.content_node))
    if edge is None or edge.belief is None or edge.cred <= 0.0:
        return None
    return VALUE_BY_CONTENT_NODE.get(record.content_node)


def _depth_one_targets(branches: Iterable[Branch]) -> List[str]:
    """Return the mandatory target layer actually traversed by Definition 4."""

    return sorted({branch.node for branch in branches if branch.depth == 1})


def _evaluate_projected_branches(
    lewm: LEWM,
    branches: Iterable[Branch],
    o_star: str,
    s1: int,
    config: SimulationConfig,
) -> Tuple[Dict[str, Result], Dict[str, float]]:
    """Evaluate deeper projections and their coherence with the parent history."""

    results: Dict[str, Result] = {}
    coherence: Dict[str, float] = {}
    for branch in sorted(branches, key=lambda item: item.path):
        if branch.depth <= 1:
            continue
        node = lewm.agents[branch.node]
        prior = _record_value(lewm, "PROJECTION", branch.node)
        result = backward_inference(
            node,
            None,
            o_star,
            Mode(),
            s1,
            config,
            focal=branch.parent or lewm.focal,
            prior_value=prior,
        )
        results[branch.node] = result
        parent = lewm.agents[branch.parent] if branch.parent is not None else None
        claims = [] if parent is None else [
            source.claim for source in parent.history
            if source.bears_on == o_star
            and source.speaker == branch.represented_agent
        ]
        coherence[branch.node] = (
            result.confidence if not claims or result.value in claims else 0.0
        )
    return results, coherence


def _evaluate_targets(
    lewm: LEWM,
    branches: Iterable[Branch],
    behaviors: Dict[str, str],
    o_star: str,
    s1: int,
    config: SimulationConfig,
    timestamp: float,
    trace: Optional[InferenceTrace],
) -> Tuple[
    Dict[str, Result], Dict[str, Result], Dict[str, Result],
    Dict[str, float], Dict[str, float],
]:
    """Traverse the current tree and compute BI/SP coherence per target."""

    results: Dict[str, Result] = {}
    pair: Dict[str, float] = {}
    projection_results: Dict[str, Result] = {}
    branch_results, branch_coherence = _evaluate_projected_branches(
        lewm, branches, o_star, s1, config
    )
    for target_key in _depth_one_targets(branches):
        prior_bi = _record_value(lewm, "BI", target_key)
        result = backward_inference(
            lewm.agents[target_key],
            behaviors[target_key],
            o_star,
            Mode(),
            s1,
            config,
            trace=trace,
            focal=lewm.focal,
            prior_value=prior_bi,
        )
        prior_sp = _record_value(lewm, "SP", target_key)
        projection = self_projection(
            lewm,
            target_key,
            o_star,
            s1,
            config,
            timestamp,
            trace,
            prior_sp,
        )
        results[target_key] = result
        projection_results[target_key] = projection
        nested = [
            confidence for node, confidence in branch_coherence.items()
            if next(
                branch.target for branch in branches if branch.node == node
            ) == target_key
        ]
        pair[target_key] = min(
            [result.confidence, projection.confidence] + nested
        )
    return (
        results,
        projection_results,
        branch_results,
        pair,
        branch_coherence,
    )


def _joint_confidence(pair: Dict[str, float], termination: str) -> float:
    if termination == "min":
        return min(pair.values())
    if termination == "weighted_average":
        return sum(pair.values()) / len(pair)
    raise ValueError("termination must be 'min' or 'weighted_average'")


def _write_back_round(
    lewm: LEWM,
    results: Dict[str, Result],
    projection_results: Dict[str, Result],
    branch_results: Dict[str, Result],
    branches: Iterable[Branch],
    timestamp: float,
) -> None:
    """Write selected content edges and separate inference records into W."""

    for target in sorted(results):
        result = results[target]
        _write_back_selected_belief(
            lewm,
            "BI",
            target,
            target,
            result,
            timestamp,
            lewm.edges[(lewm.focal, target)].obs,
        )
        projection = projection_results[target]
        projected = _sp_key(lewm.focal, target)
        _write_back_selected_belief(
            lewm,
            "SP",
            target,
            projected,
            projection,
            timestamp,
            lewm.edges[(target, projected)].obs,
        )
    branch_by_node = {branch.node: branch for branch in branches}
    for node in sorted(branch_results):
        branch = branch_by_node[node]
        _write_back_selected_belief(
            lewm,
            "PROJECTION",
            node,
            node,
            branch_results[node],
            timestamp,
            lewm.edges[(branch.parent, node)].obs,
        )
    lewm.agents[lewm.focal].timestamp = timestamp
    lewm.validate()


def mutual_reconciliation(
    lewm: LEWM,
    targets: Iterable[str],
    behaviors: Dict[str, str],
    o_star: str,
    s1: int,
    termination: str,
    t: float,
    config: SimulationConfig,
    trace: Optional[InferenceTrace] = None,
    projection_templates: Optional[
        Mapping[str, Sequence[ProjectionTemplate]]
    ] = None,
) -> Outcome:
    """Iterate over the proliferated tree with explicit round state transitions."""

    target_keys = sorted(targets)
    initial_confidence = {}
    for target in target_keys:
        record = lewm.inference_records.get(("BI", target))
        initial_confidence[target] = 1.0 if record is None else record.confidence
    branches = rebuild_and_prune(
        lewm,
        target_keys,
        initial_confidence,
        s1,
        config,
        projection_templates,
        t,
    )
    rounds: List[RoundRecord] = []
    for round_index in range(1, s1 + 1):
        round_time = t + (round_index - 1) / max(1, s1)
        sync_credibilities(lewm, round_time, config)
        (
            results,
            projection_results,
            branch_results,
            pair,
            branch_coherence,
        ) = _evaluate_targets(
            lewm, branches, behaviors, o_star, s1, config, round_time, trace
        )
        joint = _joint_confidence(pair, termination)
        _write_back_round(
            lewm,
            results,
            projection_results,
            branch_results,
            branches,
            round_time,
        )
        rounds.append(RoundRecord(
            round_index,
            round_time,
            tuple(branches),
            dict(results),
            dict(pair),
            {
                target: projection_results[target].confidence
                for target in sorted(projection_results)
            },
            dict(projection_results),
            dict(branch_results),
            dict(branch_coherence),
        ))
        if joint > config.tau_high:
            return Outcome(
                "ACCEPTED",
                joint,
                pair,
                results,
                tuple(rounds),
                projection_results,
                round_time,
            )
        if round_index >= s1:
            return Outcome(
                classify(joint, config),
                joint,
                pair,
                results,
                tuple(rounds),
                projection_results,
                round_time,
            )
        branches = rebuild_and_prune(
            lewm,
            target_keys,
            pair,
            s1,
            config,
            projection_templates,
            round_time,
        )
    raise AssertionError("unreachable reconciliation termination")
