"""Check whether local benchmark artifacts cover the paper execution plan."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CoverageRequirement:
    scenario: str
    mode: str
    libraries: tuple[str, ...]
    needs_pyperf: bool = False
    optional_libraries: tuple[str, ...] = ()
    pipeline_scopes: tuple[str, ...] = ()

    @property
    def relative_dir(self) -> Path:
        return Path(self.scenario) / self.mode


CORE_REQUIREMENTS: tuple[CoverageRequirement, ...] = (
    CoverageRequirement(
        scenario="image-rgb",
        mode="micro",
        libraries=("albumentationsx", "torchvision", "kornia", "pillow"),
        needs_pyperf=True,
    ),
    CoverageRequirement(
        scenario="image-9ch",
        mode="micro",
        libraries=("albumentationsx", "torchvision", "kornia"),
        needs_pyperf=True,
    ),
    CoverageRequirement(
        scenario="image-rgb",
        mode="pipeline",
        libraries=("albumentationsx", "torchvision", "kornia", "pillow"),
        pipeline_scopes=("memory_dataloader_augment", "decode_dataloader_augment"),
    ),
    CoverageRequirement(
        scenario="image-9ch",
        mode="pipeline",
        libraries=("albumentationsx", "torchvision", "kornia"),
        pipeline_scopes=("memory_dataloader_augment", "decode_dataloader_augment"),
    ),
    CoverageRequirement(
        scenario="video-16f",
        mode="micro",
        libraries=("albumentationsx", "torchvision", "kornia"),
        needs_pyperf=True,
    ),
    CoverageRequirement(
        scenario="video-16f",
        mode="pipeline",
        libraries=("albumentationsx", "torchvision", "kornia"),
        optional_libraries=("dali",),
        pipeline_scopes=("memory_dataloader_augment", "decode_dataloader_augment"),
    ),
)

RAM_REDUCED_REQUIREMENTS: tuple[CoverageRequirement, ...] = (
    CoverageRequirement(
        scenario="image-rgb",
        mode="micro",
        libraries=("albumentationsx", "torchvision", "kornia", "pillow"),
        needs_pyperf=True,
    ),
    CoverageRequirement(
        scenario="image-9ch",
        mode="micro",
        libraries=("albumentationsx", "torchvision", "kornia"),
        needs_pyperf=True,
    ),
    CoverageRequirement(
        scenario="image-rgb",
        mode="pipeline",
        libraries=("albumentationsx", "torchvision", "kornia", "pillow"),
        pipeline_scopes=("memory_dataloader_augment",),
    ),
    CoverageRequirement(
        scenario="image-9ch",
        mode="pipeline",
        libraries=("albumentationsx", "torchvision", "kornia"),
        pipeline_scopes=("memory_dataloader_augment",),
    ),
)

COVERAGE_PROFILES: dict[str, tuple[CoverageRequirement, ...]] = {
    "full-paper": CORE_REQUIREMENTS,
    "ram-reduced": RAM_REDUCED_REQUIREMENTS,
}


def _candidate_dirs(results_roots: list[Path], requirement: CoverageRequirement) -> list[Path]:
    dirs: list[Path] = []
    for root in results_roots:
        direct = root / requirement.relative_dir
        if direct.is_dir():
            dirs.append(direct)
        dirs.extend(path for path in root.glob(f"**/{requirement.scenario}/{requirement.mode}") if path.is_dir())
    return sorted(set(dirs))


def _has_file(dirs: list[Path], filename: str) -> bool:
    return any((directory / filename).is_file() for directory in dirs)


def _has_pipeline_file(dirs: list[Path], library: str, scope: str) -> bool:
    pattern = f"{library}_{scope}_n*_r*_w*_b*_results.json"
    return any(any(directory.glob(pattern)) for directory in dirs)


def _summary_files(requirement: CoverageRequirement, library: str) -> list[str]:
    if not requirement.pipeline_scopes:
        return [f"{library}_{requirement.mode}_results.json"]
    scopes = ("decode_dataloader_augment",) if library == "dali" else requirement.pipeline_scopes
    return [f"{library}_{scope}_n*_r*_w*_b*_results.json" for scope in scopes]


def missing_artifacts(
    results_roots: list[Path],
    *,
    require_optional_libraries: bool = False,
    requirements: tuple[CoverageRequirement, ...] = CORE_REQUIREMENTS,
) -> list[str]:
    missing: list[str] = []
    for requirement in requirements:
        dirs = _candidate_dirs(results_roots, requirement)
        if not dirs:
            missing.append(f"{requirement.scenario}/{requirement.mode}: missing result directory")
            continue

        libraries = requirement.libraries
        if require_optional_libraries:
            libraries += requirement.optional_libraries
        for library in libraries:
            if requirement.pipeline_scopes:
                scopes = ("decode_dataloader_augment",) if library == "dali" else requirement.pipeline_scopes
                missing.extend(
                    f"{requirement.scenario}/{requirement.mode}: missing {library}_{scope}_n*_r*_w*_b*_results.json"
                    for scope in scopes
                    if not _has_pipeline_file(dirs, library, scope)
                )
            else:
                missing.extend(
                    f"{requirement.scenario}/{requirement.mode}: missing {summary_file}"
                    for summary_file in _summary_files(requirement, library)
                    if not _has_file(dirs, summary_file)
                )
            if requirement.needs_pyperf:
                pyperf_file = f"{library}_{requirement.mode}_results.pyperf.json"
                if not _has_file(dirs, pyperf_file):
                    missing.append(f"{requirement.scenario}/{requirement.mode}: missing {pyperf_file}")
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results_roots",
        nargs="*",
        type=Path,
        default=[Path("gcp_runs"), Path("output")],
        help="Result roots to scan. Each root may contain scenario/mode directories directly or nested by run.",
    )
    parser.add_argument(
        "--require-optional-libraries",
        action="store_true",
        help="Require optional libraries such as DALI in addition to the core paper set.",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(COVERAGE_PROFILES),
        default="full-paper",
        help="Coverage profile to check. Use ram-reduced for the RAM-only reduced production-path pass.",
    )
    args = parser.parse_args()

    missing = missing_artifacts(
        args.results_roots,
        require_optional_libraries=args.require_optional_libraries,
        requirements=COVERAGE_PROFILES[args.profile],
    )
    if missing:
        print(f"Missing {args.profile} benchmark artifacts:")
        for item in missing:
            print(f"- {item}")
        raise SystemExit(1)
    print(f"{args.profile} benchmark artifact coverage is complete.")


if __name__ == "__main__":
    main()
