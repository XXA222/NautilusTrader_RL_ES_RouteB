"""Seed-domain validation shared by RouteB JAX entry points."""
from __future__ import annotations

from dataclasses import asdict, dataclass

MAX_JAX_SEED_EXCLUSIVE = 1 << 32


def validate_jax_seed(seed: int, *, name: str = "seed") -> None:
    """Reject seeds that JAX's uint32 PRNG key would silently alias."""
    if type(seed) is not int or not 0 <= seed < MAX_JAX_SEED_EXCLUSIVE:
        raise ValueError(f"{name} must be an integer in [0, 2**32)")


def validate_training_seed(seed: int, steps: int, *, name: str = "seed") -> None:
    """Validate the base seed and every ``seed + step`` key used by a run."""
    validate_jax_seed(seed, name=name)
    if type(steps) is not int or steps < 1 or seed + steps >= MAX_JAX_SEED_EXCLUSIVE:
        raise ValueError(f"{name} plus training updates exceeds the JAX seed domain")


def validate_jax_seed_offset(seed: int, max_offset: int, *, name: str = "seed") -> None:
    """Validate a base seed and its largest deterministic derived offset."""
    validate_jax_seed(seed, name=name)
    if (type(max_offset) is not int or max_offset < 0
            or seed + max_offset >= MAX_JAX_SEED_EXCLUSIVE):
        raise ValueError(f"{name} plus derived offset exceeds the JAX seed domain")


@dataclass(frozen=True)
class SeedStreams:
    """Optional independent roots for controlled initialization/sampling/noise probes."""

    initialization_seed: int | None = None
    sampler_seed: int | None = None
    model_seed: int | None = None


@dataclass(frozen=True)
class ResolvedSeedStreams:
    initialization_seed: int
    sampler_seed: int
    model_seed: int


def resolve_training_seed_streams(base_seed: int, steps: int,
                                  streams: SeedStreams | None = None) -> ResolvedSeedStreams:
    """Resolve independent PRNG roots while preserving the legacy default stream.

    With no explicit plan, initialization and window sampling use ``base_seed``;
    stochastic model updates use ``base_seed + 1``. This exactly preserves the
    existing RouteB training sequence. Explicit roots support controlled,
    checkpoint-bound synthetic diagnostics and future preregistered runs.
    """
    validate_jax_seed(base_seed, name="base training seed")
    if type(steps) is not int or steps < 1:
        raise ValueError("training steps must be a positive integer")
    if streams is not None and type(streams) is not SeedStreams:
        raise ValueError("seed_streams must be a SeedStreams value")
    plan = streams or SeedStreams()
    initialization = (base_seed if plan.initialization_seed is None
                      else plan.initialization_seed)
    sampler = base_seed if plan.sampler_seed is None else plan.sampler_seed
    model = base_seed + 1 if plan.model_seed is None else plan.model_seed
    validate_jax_seed(initialization, name="initialization seed")
    validate_jax_seed_offset(sampler, steps - 1, name="sampler seed")
    validate_jax_seed_offset(model, steps - 1, name="model seed")
    return ResolvedSeedStreams(initialization, sampler, model)


def seed_stream_identity(streams: ResolvedSeedStreams) -> dict[str, int]:
    """Return a stable JSON-compatible identity for an already resolved plan."""
    if (type(streams) is not ResolvedSeedStreams
            or any(type(value) is not int for value in asdict(streams).values())):
        raise ValueError("resolved seed streams must contain concrete integer roots")
    return asdict(streams)
