from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

from benchmark.config.resolve import run_config_payload

if TYPE_CHECKING:
    from benchmark.config.models import BenchmarkRunConfig

RUN_CONFIG_ENV_VAR = "BENCHMARK_RUN_CONFIG_JSON"


def install_run_config_env(config: BenchmarkRunConfig) -> None:
    os.environ[RUN_CONFIG_ENV_VAR] = json.dumps(run_config_payload(config))


def run_config_payload_from_env() -> dict[str, Any] | None:
    raw = os.environ.get(RUN_CONFIG_ENV_VAR)
    if not raw:
        return None
    return json.loads(raw)
