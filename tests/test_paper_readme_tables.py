from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "paper"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import generate_figures_and_insights as figures  # noqa: E402


def test_readme_transform_tables_include_ci_and_bold_row_winner(tmp_path: Path, monkeypatch) -> None:
    generated = tmp_path / "generated"
    generated.mkdir()
    (generated / "all_results.csv").write_text(
        "regime,regime_label,library,transform,supported,early_stopped,"
        "num_successful_runs,median_throughput,ci95\n"
        "test_cpu,CPU,a,Affine,True,False,3,10.0,1.5\n"
        "test_gpu,GPU,b,Affine,True,False,3,12.0,0.5\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(figures, "GENERATED", generated)
    monkeypatch.setattr(
        figures,
        "README_SCENARIO_TABLES",
        [("Test", "test_", [("test_cpu", "a"), ("test_gpu", "b")])],
    )

    markdown = figures._readme_transform_tables_markdown()

    assert "Image table values are medians with 95% confidence intervals when available" in markdown
    assert "| Transform | a<br>CPU | b<br>GPU |" in markdown
    assert "| --- | ---: | ---: |" in markdown
    assert "| Affine | 10.0 ± 1.5 | **12.0 ± 0.5** |" in markdown


def test_readme_result_cell_omits_missing_ci() -> None:
    row = figures.pd.Series({"median_throughput": 8.0, "ci95": float("nan")})

    assert figures._readme_result_cell(row, bold=True) == "**8.0**"


def test_scenario_overview_uses_regime_universe_denominator(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(figures, "GENERATED", tmp_path)
    monkeypatch.setattr(figures, "_savefig", lambda *_args, **_kwargs: figures.plt.close())
    df = figures.pd.DataFrame(
        [
            {
                "regime": "video16f_dataloader_gpu",
                "regime_label": "Video GPU DataLoader",
                "library": "torchvision",
                "transform": "Affine",
                "measured": True,
                "median_throughput": 300.0,
            },
            {
                "regime": "video16f_dataloader_gpu",
                "regime_label": "Video GPU DataLoader",
                "library": "torchvision",
                "transform": "Rotate",
                "measured": True,
                "median_throughput": 100.0,
            },
            {
                "regime": "video16f_dataloader_gpu",
                "regime_label": "Video GPU DataLoader",
                "library": "pytorchvideo",
                "transform": "Canonical",
                "measured": True,
                "median_throughput": 20.0,
            },
        ],
    )

    summary = figures._scenario_overview_plot(
        df,
        scenario_prefix="video16f_",
        output_stem="video16f",
        item_unit="clips",
    )

    pytorchvideo = summary[summary["library"].astype(str) == "pytorchvideo"].iloc[0]
    assert int(pytorchvideo["full"]) == 1
    assert int(pytorchvideo["total_rows"]) == 3


def test_scenario_overview_rejects_ambiguous_regime_labels(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(figures, "GENERATED", tmp_path)
    df = figures.pd.DataFrame(
        [
            {
                "regime": "video16f_dataloader_gpu",
                "regime_label": "Video GPU DataLoader",
                "library": "torchvision",
                "transform": "Affine",
                "measured": True,
                "median_throughput": 300.0,
            },
            {
                "regime": "video16f_alternate_gpu",
                "regime_label": "Video GPU DataLoader",
                "library": "torchvision",
                "transform": "Rotate",
                "measured": True,
                "median_throughput": 100.0,
            },
        ],
    )

    with pytest.raises(ValueError, match="map to one regime"):
        figures._scenario_overview_plot(
            df,
            scenario_prefix="video16f_",
            output_stem="video16f",
            item_unit="clips",
        )


def test_validate_known_values_rejects_unknown_category() -> None:
    values = figures.pd.Series(["albumentationsx", "new-library"])

    with pytest.raises(ValueError, match="Unexpected library values: new-library"):
        figures._validate_known_values(values, figures.LIBRARY_ORDER, "library")
