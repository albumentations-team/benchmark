from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from zipfile import ZipFile

import pytest

ROOT = Path(__file__).parents[1]


def test_bootstrap_is_self_contained_and_has_valid_shell_syntax() -> None:
    script = ROOT / "infra" / "gcp" / "bootstrap.sh"
    source = script.read_text(encoding="utf-8")

    assert "ensure_uv" in source
    assert 'source "$CODE_ROOT/infra/gcp/' not in source
    assert 'exec > >(tee -a "$LOG_PATH") 2>&1' in source
    assert "logs/bootstrap-${instance_id}.log" in source
    assert "--torch-backend cu130" in source
    assert source.count('uv python install --no-bin "$python_version"') == 2
    subprocess.run(["bash", "-n", str(script)], check=True)  # noqa: S603, S607 - fixed local script path.


def test_rgb_launch_uses_the_current_bootstrap() -> None:
    source = (ROOT / "src" / "augbench" / "launch_rgb.py").read_text(encoding="utf-8")

    assert '"infra/gcp/bootstrap.sh"' in source
    assert "bootstrap_v2" not in source


@pytest.mark.parametrize("wheel_hash", ["valid", "invalid"])
@pytest.mark.parametrize("cached_version", ["1.0", "2.0"])
def test_cached_environment_is_synced_to_the_verified_lock(
    tmp_path: Path, wheel_hash: str, cached_version: str
) -> None:
    uv = shutil.which("uv")
    assert uv is not None
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    cached = _write_wheel(wheels, "cache_probe", cached_version)
    current = _write_wheel(wheels, "cache_probe", "2.0")
    extra = _write_wheel(wheels, "cache_extra", "1.0")
    environment = tmp_path / "cached"
    env = {
        **os.environ,
        "UV_CACHE_DIR": str(tmp_path / "uv-cache"),
        "UV_OFFLINE": "1",
        "UV_NO_INDEX": "1",
        "UV_FIND_LINKS": str(wheels),
    }
    subprocess.run([uv, "venv", "--python", sys.executable, str(environment)], env=env, check=True)  # noqa: S603
    subprocess.run(  # noqa: S603
        [uv, "pip", "install", "--python", str(environment / "bin/python"), str(cached), str(extra)],
        env=env,
        check=True,
    )
    cached_module = next(environment.glob("lib/python*/site-packages/cache_probe.py"))
    cached_module.write_text("VERSION = 'corrupted'\n")
    cache_archive = tmp_path / "cached.tar.gz"
    with tarfile.open(cache_archive, "w:gz") as archive:
        archive.add(environment, arcname=".")
    lock_path = tmp_path / "lock.txt"
    digest = hashlib.sha256(current.read_bytes()).hexdigest() if wheel_hash == "valid" else "0" * 64
    lock_path.write_text(f"cache-probe==2.0 --hash=sha256:{digest}\n")
    state_root = tmp_path / "state"
    (state_root / "downloads").mkdir(parents=True)
    source = (ROOT / "infra/gcp/bootstrap.sh").read_text()
    functions = source[source.index("repair_python_links() {") : source.index("\npublish_log() {")]
    script = functions
    if sys.platform == "darwin":
        script += '\nsha256sum() { shasum -a 256 "$@"; }\n'
    script += """
uv() {
  case "$1 $2" in
    'python install') return 0 ;;
    'python find') printf '%s\\n' "$TEST_PYTHON" ;;
    *) "$TEST_UV" "$@" ;;
  esac
}
gcloud() { cp "$TEST_CACHE" "${@: -1}"; }
stage_environment "$@"
"""
    result = subprocess.run(  # noqa: S603 - executes bootstrap functions with local tool fixtures.
        [
            "/bin/bash",
            "-euc",
            script,
            "test",
            "gs://cache/environment.tar.gz",
            str(lock_path),
            hashlib.sha256(lock_path.read_bytes()).hexdigest(),
            "3.13.14",
        ],
        env={
            **env,
            "STATE_ROOT": str(state_root),
            "TEST_UV": uv,
            "TEST_PYTHON": sys._base_executable,
            "TEST_CACHE": str(cache_archive),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    if wheel_hash == "invalid":
        assert result.returncode != 0
        assert "hash mismatch" in result.stderr.lower()
        return
    assert result.returncode == 0, result.stderr
    restored = subprocess.check_output(  # noqa: S603 - runs the virtual environment built in this test.
        [
            str(state_root / "environment/bin/python"),
            "-c",
            (
                "import cache_probe; from importlib.metadata import distributions; "
                "print(cache_probe.VERSION); print(sorted(d.metadata['Name'] for d in distributions()))"
            ),
        ],
        text=True,
    )
    assert restored.splitlines() == ["2.0", "['cache_probe']"]


def _write_wheel(directory: Path, name: str, version: str) -> Path:
    path = directory / f"{name}-{version}-py3-none-any.whl"
    metadata = f"{name}-{version}.dist-info"
    files = {
        f"{name}.py": f"VERSION = {version!r}\n",
        f"{metadata}/METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
        f"{metadata}/WHEEL": "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    files[f"{metadata}/RECORD"] = "".join(f"{filename},,\n" for filename in [*files, f"{metadata}/RECORD"])
    with ZipFile(path, "w") as wheel:
        for filename, content in files.items():
            wheel.writestr(filename, content)
    return path
