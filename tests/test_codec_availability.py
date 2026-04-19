"""Availability probes for each registered preset (pydub round-trip only)."""

from __future__ import annotations

import pytest

from asv.codec import check_preset_available, clear_availability_cache, list_presets


def setup_function() -> None:
    clear_availability_cache()


@pytest.mark.parametrize("name", list_presets())
def test_each_preset_check_returns_bool(name: str) -> None:
    clear_availability_cache()
    ok = check_preset_available(name, use_cache=False)
    assert isinstance(ok, bool)
