from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_bootstrap_is_self_contained_and_has_valid_shell_syntax() -> None:
    script = ROOT / "infra" / "gcp" / "bootstrap.sh"
    source = script.read_text(encoding="utf-8")

    assert "ensure_uv" in source
    assert 'source "$CODE_ROOT/infra/gcp/' not in source
    assert 'exec > >(tee -a "$LOG_PATH") 2>&1' in source
    assert "logs/bootstrap-${instance_id}.log" in source
    assert "--torch-backend cu130" in source
    subprocess.run(["bash", "-n", str(script)], check=True)  # noqa: S603, S607 - fixed local script path.


def test_rgb_launch_uses_the_current_bootstrap() -> None:
    source = (ROOT / "src" / "augbench" / "launch_rgb.py").read_text(encoding="utf-8")

    assert '"infra/gcp/bootstrap.sh"' in source
    assert "bootstrap_v2" not in source
