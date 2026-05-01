from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURE_MODULES = (
    "benchmark/cli.py",
    "benchmark/matrix.py",
    "benchmark/policy.py",
    "benchmark/jobs.py",
    "benchmark/orchestrator.py",
    "benchmark/envs.py",
    "benchmark/specs/load.py",
    "benchmark/media/loaders.py",
    "benchmark/pyperf_micro_runner.py",
    "benchmark/pipeline_runner.py",
    "benchmark/runner.py",
)


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_architecture_doc_references_existing_core_modules() -> None:
    doc = _read("docs/benchmark_architecture.md")

    for module_path in ARCHITECTURE_MODULES:
        assert module_path in doc
        assert (REPO_ROOT / module_path).exists()


def test_scope_and_readme_link_to_architecture_doc() -> None:
    assert "docs/benchmark_architecture.md" in _read("README.md")
    assert "docs/benchmark_architecture.md" in _read("docs/benchmark_scope.md")


def test_skills_document_centralized_policy_and_matrix() -> None:
    skill_paths = (
        ".cursor/skills/benchmark-runner/SKILL.md",
        ".cursor/skills/documentation-generator/SKILL.md",
        ".cursor/skills/library-integration/SKILL.md",
        ".cursor/skills/paper-benchmark-execution/SKILL.md",
    )
    required_refs = (
        "benchmark/matrix.py",
        "benchmark/policy.py",
        "benchmark/jobs.py",
        "benchmark/orchestrator.py",
    )

    for skill_path in skill_paths:
        text = _read(skill_path)
        for ref in required_refs:
            assert ref in text, f"{skill_path} must mention {ref}"


def test_fair_fast_rule_blocks_cli_backend_drift() -> None:
    rule = _read(".cursor/rules/benchmark_fair_fast.mdc")

    assert "benchmark/matrix.py" in rule
    assert "benchmark/policy.py" in rule
    assert "benchmark/jobs.py" in rule
    assert "benchmark/orchestrator.py" in rule
    assert "Do not reintroduce special backend branches in `benchmark/cli.py`" in rule
