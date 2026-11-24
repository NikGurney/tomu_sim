"""Six executable residue boundary checks of Definition 5 (manuscript Appendix A.3, Table A2)."""

import random
from typing import Callable, List, Tuple

from runs.common import aligned_table
from tomu.config import DEFAULT_CONFIG, SimulationConfig
from tomu.lewm import Edge
from tomu.residue import accrue_residue, credibility, rehabilitate, residue


def _edge() -> Edge:
    """Return a fresh synthetic edge with no residue."""

    return Edge("a1", "a2", None, 1.0, 1.0)


def check_targeting(config: SimulationConfig = DEFAULT_CONFIG) -> None:
    """Assert an untouched edge retains zero residue and full credibility."""

    edge = _edge()
    assert residue(edge, 10.0, config) == 0.0 and credibility(edge, 10.0, config) == 1.0


def check_disconfirmation(config: SimulationConfig = DEFAULT_CONFIG) -> None:
    """Assert larger disconfirmation creates larger residue."""

    weak, strong = _edge(), _edge()
    accrue_residue(weak, 0.2, 1.0, config)
    accrue_residue(strong, 0.9, 1.0, config)
    assert residue(strong, 1.0, config) > residue(weak, 1.0, config)


def check_decay(config: SimulationConfig = DEFAULT_CONFIG) -> None:
    """Assert residue samples are non-increasing between events."""

    edge = _edge()
    accrue_residue(edge, 0.8, 1.0, config)
    samples = [residue(edge, t, config) for t in [1.0, 2.0, 5.0, 20.0]]
    assert all(later <= earlier for earlier, later in zip(samples, samples[1:]))


def check_positive_credibility(config: SimulationConfig = DEFAULT_CONFIG) -> None:
    """Assert credibility stays positive after a finite rejection sequence."""

    edge = _edge()
    for event in range(1, 101):
        accrue_residue(edge, 0.9, float(event), config)
    assert credibility(edge, 100.0, config) > 0.0


def check_floor(config: SimulationConfig = DEFAULT_CONFIG) -> None:
    """Assert time and full acute rehabilitation never cross the floor."""

    edge = _edge()
    accrue_residue(edge, 1.0, 1.0, config)
    for event in range(2, 5):
        rehabilitate(edge, 1.0, float(event), config)
    later = [residue(edge, t, config) for t in [4.0, 20.0, 100.0]]
    assert edge.floor > 0.0 and all(value >= edge.floor for value in later) and later[0] == edge.floor


def check_rehabilitation(config: SimulationConfig = DEFAULT_CONFIG) -> None:
    """Assert one supporting signal lowers residue but not below the floor."""

    edge = _edge()
    accrue_residue(edge, 1.0, 1.0, config)
    before = residue(edge, 1.0, config)
    rehabilitate(edge, 0.5, 1.0, config)
    after = residue(edge, 1.0, config)
    assert edge.floor <= after < before


def evaluate_checks() -> List[Tuple[str, bool]]:
    """Run all six checks and return a stable PASS/FAIL record."""

    checks: List[Tuple[str, Callable[[], None]]] = [
        ("Targeting", check_targeting),
        ("More disconfirmation", check_disconfirmation),
        ("Decay over time", check_decay),
        ("Credibility positive", check_positive_credibility),
        ("Persistence floor", check_floor),
        ("Partial rehabilitation", check_rehabilitation),
    ]
    results: List[Tuple[str, bool]] = []
    for name, check in checks:
        try:
            check()
            results.append((name, True))
        except AssertionError:
            results.append((name, False))
    return results


def main() -> None:
    """Print the six-row table and exit non-zero through an assertion on failure."""

    random.seed(DEFAULT_CONFIG.seed)
    results = evaluate_checks()
    print(aligned_table(["property", "result"], [[name, "PASS" if passed else "FAIL"] for name, passed in results]))
    failures = [name for name, passed in results if not passed]
    assert not failures, f"residue checks failed: {failures}"


if __name__ == "__main__":
    main()
