# ruff: noqa: INP001
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Mapping


PALETTE = {
    "albumentationsx": "#177245",
    "torchvision": "#2f6fbd",
    "kornia": "#8a4fb5",
    "pillow": "#c47a1b",
    "dali": "#5f6b2f",
}
LIBRARY_DISPLAY = {
    "albumentationsx": "AlbumentationsX",
    "torchvision": "TorchVision",
    "kornia": "Kornia",
    "pillow": "Pillow",
    "dali": "DALI",
}
LIBRARY_ORDER = ["albumentationsx", "torchvision", "kornia", "pillow", "dali"]
REGIME_LABELS = {
    "rgb_micro_cpu": "CPU micro",
    "rgb_micro_gpu": "GPU micro",
    "rgb_dataloader_cpu": "CPU DataLoader",
    "rgb_dataloader_gpu": "GPU DataLoader",
    "image9ch_micro_cpu": "9ch CPU micro",
    "image9ch_micro_gpu": "9ch GPU micro",
    "image9ch_dataloader_cpu": "9ch CPU DataLoader",
    "image9ch_dataloader_gpu": "9ch GPU DataLoader",
}
REGIME_ORDER = [
    "CPU micro",
    "CPU DataLoader",
    "GPU micro",
    "GPU DataLoader",
    "9ch CPU micro",
    "9ch CPU DataLoader",
    "9ch GPU micro",
    "9ch GPU DataLoader",
]


class MeasuredRow(Protocol):
    supported: bool
    early_stopped: bool
    num_successful_runs: int


def fmt(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "-"
    if not math.isfinite(value):
        return "-"
    return f"{value:.{digits}f}"


def fmt_ratio(value: float) -> str:
    if not math.isfinite(value):
        return "-"
    return f"{value:.2f}x"


def implementation_label(regime: str, library: str) -> str:
    device = "GPU" if regime.endswith("_dataloader_gpu") else "CPU"
    return f"{LIBRARY_DISPLAY.get(library, library)} {device}"


def is_measured(row: MeasuredRow) -> bool:
    return row.supported and not row.early_stopped and row.num_successful_runs > 0


def latex_escape(value: str) -> str:
    replacements: Mapping[str, str] = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def recipe_display_name(recipe: str) -> str:
    label = recipe
    suffix = "+Normalize+ToTensor"
    label = label.removesuffix(suffix)
    prefix = "RandomCrop224+"
    label = label.removeprefix(prefix)
    return label or "RandomCrop224"
