from __future__ import annotations

import hashlib
import logging
import subprocess
import sys
from typing import TYPE_CHECKING

from benchmark.matrix import library_env_group, requirements_for_env_group

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)
REQUIREMENTS_REFRESH_TIMEOUT_SECONDS = 180


def venv_python(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def requirement_input_path(requirements_path: Path) -> Path:
    return requirements_path.with_suffix(".in")


def compile_requirements(python: Path, requirements_path: Path) -> None:
    requirements_input = requirement_input_path(requirements_path)
    if not requirements_input.exists():
        return
    logger.info("Refreshing %s from %s", requirements_path, requirements_input)
    try:
        subprocess.run(  # noqa: S603 - python executable and requirements path come from the resolved benchmark env.
            [
                str(python),
                "-m",
                "uv",
                "pip",
                "compile",
                "--upgrade",
                "--quiet",
                "-o",
                str(requirements_path),
                str(requirements_input),
            ],
            check=True,
            timeout=REQUIREMENTS_REFRESH_TIMEOUT_SECONDS,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        if requirements_path.exists():
            logger.warning("Could not refresh %s (%s); keeping existing lock", requirements_path, e)
            return
        raise


def requirements_cache_key(
    *,
    python: Path,
    requirements_paths: list[Path],
    env_group: str,
    media: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(f"python={python}\n".encode())
    try:
        python_version = subprocess.check_output(  # noqa: S603 - local interpreter version probe.
            [str(python), "--version"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        python_version = "unknown"
    digest.update(f"python_version={python_version}\n".encode())
    digest.update(f"env_group={env_group}\nmedia={media}\n".encode())
    for path in requirements_paths:
        digest.update(f"path={path.name}\n".encode())
        digest.update(path.read_bytes())
        digest.update(b"\n")
    return digest.hexdigest()


def ensure_uv(python: Path) -> None:
    try:
        subprocess.run(  # noqa: S603 - controlled module invocation in benchmark venv.
            [str(python), "-m", "uv", "--version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        logger.info("Installing uv into %s", python.parent.parent)
        subprocess.run([str(python), "-m", "pip", "install", "-q", "uv"], check=True)  # noqa: S603


def install_requirements_if_needed(
    *,
    python: Path,
    venv_dir: Path,
    requirements_paths: list[Path],
    env_group: str,
    media: str,
) -> None:
    cache_file = venv_dir / ".benchmark_requirements_hash"
    cache_key = requirements_cache_key(
        python=python,
        requirements_paths=requirements_paths,
        env_group=env_group,
        media=media,
    )
    if cache_file.exists() and cache_file.read_text(encoding="utf-8").strip() == cache_key:
        logger.info("Using cached dependencies for %s (%s)", env_group, media)
        return

    logger.info("Installing dependencies for %s (%s)...", env_group, media)
    for requirements_path in requirements_paths:
        subprocess.run(  # noqa: S603 - requirements are resolved from the benchmark matrix.
            [str(python), "-m", "uv", "pip", "install", "-q", "-U", "-r", str(requirements_path)],
            check=True,
        )
    cache_file.write_text(f"{cache_key}\n", encoding="utf-8")
    logger.info("Dependencies ready for %s", env_group)


def ensure_venv(library: str, media: str, repo_root: Path, *, refresh_requirements: bool = True) -> Path:
    env_group = library_env_group(library, media)
    venv_dir = repo_root / f".venv_{env_group}"

    if not venv_dir.exists():
        logger.info("Creating venv for %s (%s)...", env_group, media)
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)  # noqa: S603

    python = venv_python(venv_dir)
    ensure_uv(python)

    requirements_paths = requirements_for_env_group(env_group, media, repo_root)
    if refresh_requirements:
        for requirements_path in requirements_paths:
            compile_requirements(python, requirements_path)

    install_requirements_if_needed(
        python=python,
        venv_dir=venv_dir,
        requirements_paths=requirements_paths,
        env_group=env_group,
        media=media,
    )

    return python
