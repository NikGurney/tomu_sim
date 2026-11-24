"""Central configuration for every tunable (manuscript Appendix A.1, Table A1)."""

from dataclasses import dataclass, field
from typing import Dict, Tuple

# --- Sophistication (Model, Def 2 & 3) ---
S_MAX = 6
S1_DEFAULT = 3

# --- Proliferation (Def 3) ---
RHO = 0.5
KAPPA = 0.05

# --- Inference thresholds (Def 4) ---
TAU_HIGH = 0.90
TAU_LOW = 0.50
TAU_COMMIT = 0.70
DELTA_T_SESSION = 1.0
PICKLE_BLS_SCALARS = (0.6, 0.8, 1.0)

# --- Fit / source weighting (Appendix A.1) ---
BELIEF_SOURCE_STRENGTH = 0.5
UNTYPED_STRENGTH = 0.5
DISCOUNT_FACTOR = 0.1

# --- Residue (Definition 5; Appendix A.3, Eq. A1-A2) ---
LAMBDA = 0.1
ALPHA = 0.5
BETA = 0.2
GAMMA = 0.3

# --- Reproducibility ---
SEED = 0

# --- Run 2 scenario (Appendix A.4.2) ---
RUN2_INVOCATIONS = 20
RUN2_TARGET_OBS = {"a2": 0.55, "a3": 0.20, "a4": 0.40}

BELIEF_VALUES = ["absent", "chocolate", "pickle", "popcorn"]
BEHAVIORS = ["NEUTRAL", "CONTENT"]
CONTENT_NODE_BY_VALUE = {
    "chocolate": "o_chocolate",
    "pickle": "o_pickle",
    "popcorn": "o_popcorn",
}


@dataclass(frozen=True)
class SimulationConfig:
    """Immutable parameter snapshot passed explicitly through computations."""

    s_max: int = S_MAX
    s1_default: int = S1_DEFAULT
    rho: float = RHO
    kappa: float = KAPPA
    tau_high: float = TAU_HIGH
    tau_low: float = TAU_LOW
    tau_commit: float = TAU_COMMIT
    delta_t_session: float = DELTA_T_SESSION
    pickle_bls_scalars: Tuple[float, ...] = PICKLE_BLS_SCALARS
    belief_source_strength: float = BELIEF_SOURCE_STRENGTH
    untyped_strength: float = UNTYPED_STRENGTH
    discount_factor: float = DISCOUNT_FACTOR
    decay_rate: float = LAMBDA
    alpha: float = ALPHA
    beta: float = BETA
    gamma: float = GAMMA
    seed: int = SEED
    run2_invocations: int = RUN2_INVOCATIONS
    run2_target_obs: Dict[str, float] = field(
        default_factory=lambda: dict(RUN2_TARGET_OBS)
    )


DEFAULT_CONFIG = SimulationConfig()
