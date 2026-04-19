"""Validation utilities for ASV training."""

from asv.validation.voxceleb import (
    VoxCelebEERValidator,
    VoxCelebVariantEERValidator,
    collect_variant_validation_roots,
    compute_eer,
    parse_trial_setups,
    resolve_variant_validation_trial_files,
)

__all__ = [
    "VoxCelebEERValidator",
    "VoxCelebVariantEERValidator",
    "collect_variant_validation_roots",
    "compute_eer",
    "parse_trial_setups",
    "resolve_variant_validation_trial_files",
]