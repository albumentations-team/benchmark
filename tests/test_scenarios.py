from __future__ import annotations

import pytest

from benchmark.scenarios import get_scenario, resolve_decoders, resolve_libraries, resolve_mode


def test_video_decode_scenario_defaults_to_decode() -> None:
    scenario = get_scenario("video-decode-16f")
    assert resolve_mode(scenario, None) == "decode"
    assert scenario.clip_length == 16


def test_dali_available_for_pipeline_libraries_only() -> None:
    video = get_scenario("video-16f")
    rgb = get_scenario("image-rgb")

    assert "dali" not in resolve_libraries(video, "micro", None)
    assert "dali" in resolve_libraries(video, "pipeline", None)
    assert "dali_experimental" not in resolve_libraries(video, "micro", None)
    assert "dali_experimental" in resolve_libraries(video, "pipeline", None)
    assert "pytorchvideo" not in resolve_libraries(video, "micro", None)
    assert "pytorchvideo" in resolve_libraries(video, "pipeline", None)
    assert "dali" not in resolve_libraries(rgb, "micro", None)
    assert "dali" in resolve_libraries(rgb, "pipeline", None)


def test_unknown_library_lists_available() -> None:
    scenario = get_scenario("image-rgb")
    with pytest.raises(ValueError, match="Available"):
        resolve_libraries(scenario, "micro", ["missing"])


def test_decode_scenario_rejects_unknown_decoder() -> None:
    scenario = get_scenario("video-decode-16f")
    with pytest.raises(ValueError, match="Available"):
        resolve_decoders(scenario, ["missing"])
