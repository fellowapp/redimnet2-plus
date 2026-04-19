"""Сборка :class:`SpecAugmentConfig` из Hydra-конфига для трейн-цикла.

Читает настройки спектральных аугментаций (freq/time mask, cutmix) сначала из
``augment.feature_levels`` (предпочтительное место, см. ``configs/augment/default.yaml``),
если там нет — из legacy-места ``data.spec_augment``."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

from asv.data import SpecAugmentConfig


def build_spec_augment_cfg(cfg: DictConfig) -> SpecAugmentConfig:
    """Собирает :class:`SpecAugmentConfig` из конфига аугментаций.

    Предпочтительное место:
      - ``augment.feature_levels[*].name == "spec_augment"``: используются вложенные
        параметры ``spec_augment``, флаг ``one_of`` с уровня и ``p`` уровня как
        :attr:`SpecAugmentConfig.level_p` (вероятность применить что-либо на батче).

    Обратно-совместимый fallback:
      - ``data.spec_augment`` (опционально ``one_of``, ``level_p`` и поддеревья
        mask/cutmix)."""
    sa: dict[str, Any] | DictConfig = {}
    one_of = False
    level_p = 1.0
    for level in cfg.augment.get("feature_levels", []):
        if str(level.get("name", "")) == "spec_augment":
            sa = level.get("spec_augment", {})
            one_of = bool(level.get("one_of", False))
            level_p = float(level.get("p", 1.0))
            break
    if not sa:
        sa = cfg.data.get("spec_augment", {})
        one_of = bool(sa.get("one_of", False))
        level_p = float(sa.get("level_p", 1.0))
    freq = sa.get("freq_mask", {})
    time = sa.get("time_mask", {})
    cm = sa.get("cutmix", {})
    return SpecAugmentConfig(
        one_of=one_of,
        level_p=level_p,
        freq_mask_p=float(freq.get("p", 0.5)),
        freq_mask_max_width=int(freq.get("max_width", 8)),
        time_mask_p=float(time.get("p", 0.5)),
        time_mask_max_width=int(time.get("max_width", 10)),
        cutmix_p=float(cm.get("p", 0.3)),
        cutmix_min_frames=int(cm.get("min_frames", 5)),
        cutmix_max_frames=int(cm.get("max_frames", 30)),
    )
