from __future__ import annotations

import sys
from pathlib import Path

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
