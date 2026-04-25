from __future__ import annotations

import pytest

from benchmark.scenarios import get_scenario, resolve_decoders, resolve_libraries, resolve_mode


def test_video_decode_scenario_defaults_to_decode() -> None:
    scenario = get_scenario("video-decode-16f")
    assert resolve_mode(scenario, None) == "decode"
    assert scenario.clip_length == 16


def test_dali_only_available_for_video_pipeline_libraries() -> None:
    scenario = get_scenario("video-16f")
    assert "dali" not in resolve_libraries(scenario, "micro", None)
    assert "dali" in resolve_libraries(scenario, "pipeline", None)


def test_unknown_library_lists_available() -> None:
    scenario = get_scenario("image-rgb")
    with pytest.raises(ValueError, match="Available"):
        resolve_libraries(scenario, "micro", ["missing"])


def test_decode_scenario_rejects_unknown_decoder() -> None:
    scenario = get_scenario("video-decode-16f")
    with pytest.raises(ValueError, match="Available"):
        resolve_decoders(scenario, ["missing"])
