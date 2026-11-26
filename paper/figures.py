# ruff: noqa: INP001
"""PGFPlots sources for the report figures; the paper inputs them and the README renders them to PNG."""

from __future__ import annotations

from statistics import median
from typing import TYPE_CHECKING

from paper.intervals import RatioInterval, ratio_interval

if TYPE_CHECKING:
    from collections.abc import Sequence

    from paper.generate_results import _PaperData
    from paper.generate_training import Row

LABELS = {
    "albumentationsx_cpu": "AlbumentationsX",
    "pillow_cpu": "Pillow",
    "pillow_allsimd_cpu": "Pillow-AllSIMD",
    "torchvision_cpu": "TorchVision CPU",
    "torchvision_gpu": "TorchVision GPU",
    "kornia_cpu": "Kornia CPU",
    "kornia_gpu": "Kornia GPU",
    "dali_gpu": "DALI",
}
_AX = "albumentationsx_cpu"
_DOT = " \\textperiodcentered{} "
_STYLE = (
    "axis lines*=left, y axis line style={draw=none}, ytick style={draw=none}, xmajorgrids, "
    "major grid style={black!10}, tick label style={font=\\small}, label style={font=\\small}, clip=false"
)
# Okabe-Ito hues stay distinguishable for the common color-vision deficiencies; each also gets its own mark.
_RECIPE_MARKS = (
    ("recipeA", "*"),
    ("recipeB", "square*"),
    ("recipeC", "diamond*"),
    ("recipeD", "triangle*"),
)


def _coordinates(points: Sequence[tuple[float, float]]) -> str:
    return " ".join(f"({x:.6g},{y:.4g})" for x, y in points)


def _scatter(style: str, points: Sequence[tuple[float, float]]) -> str:
    return f"\\addplot[only marks, {style}] coordinates {{{_coordinates(points)}}};"


def _rows(labels: Sequence[str]) -> str:
    return f"ytick={{0,...,{len(labels) - 1}}}, yticklabels={{{','.join(labels)}}}, ymin=-0.7, ymax={len(labels) - 0.3}"


def rgb_overview(data: _PaperData) -> str:
    """Images per second on every recipe all paths share, one dot per recipe, with each path's median."""
    paths = [implementation for implementation, _ in data.implementations]
    medians = {path: median(data.speed[path, recipe] for recipe in data.common) for path in paths}
    order = sorted(paths, key=medians.__getitem__)
    recipes = [(order.index(path), data.speed[path, recipe]) for path in order for recipe in data.common]
    memory = {path: median(data.memory[path, recipe] for recipe in data.common) for path in paths}
    labels = [LABELS[path] for path in order]
    dots = _scatter("mark=*, mark size=1.2pt, draw=none, fill=black!40, fill opacity=0.6", [(x, y) for y, x in recipes])
    marks = _scatter(
        "mark=|, mark size=5pt, line width=1.8pt, draw=black", [(medians[path], row) for row, path in enumerate(order)]
    )
    memory_marks = _scatter(
        "mark=*, mark size=1.8pt, draw=black!75, fill=black!75", [(memory[path], row) for row, path in enumerate(order)]
    )
    value_nodes = "\n".join(
        f"\\node[anchor=west,font=\\footnotesize] at (axis cs:9000,{row}) {{{medians[path]:,.0f}}};"
        for row, path in enumerate(order)
    )
    return f"""\\begin{{tikzpicture}}
\\begin{{axis}}[name=speed, width=9.2cm, height=6.6cm, xmode=log, xmin=400, xmax=8000,
  xtick={{500,1000,2000,4000,8000}}, xticklabels={{500,1000,2000,4000,8000}}, {_rows(labels)},
  xlabel={{Images per second (log scale)}}, {_STYLE}]
{dots}
{marks}
{value_nodes}
\\end{{axis}}
\\begin{{axis}}[at={{(speed.east)}}, anchor=west, xshift=1.5cm, width=4.4cm, height=6.6cm, xmin=0, xmax=2000,
  xtick={{0,1000,2000}}, ytick=\\empty, ymin=-0.7, ymax={len(order) - 0.3}, xlabel={{Peak GPU memory (MiB)}}, {_STYLE}]
{memory_marks}
\\end{{axis}}
\\end{{tikzpicture}}
"""


def ax_pairs(data: _PaperData, paths: Sequence[str] | None = None) -> str:
    """Every shared recipe's throughput ratio against AX, marked by what the seed intervals can tell."""
    if paths is None:
        paths = [implementation for implementation, _ in data.implementations if implementation != _AX]
    groups: dict[str, list[tuple[float, float]]] = {"faster": [], "indistinguishable": [], "slower": []}
    counts: dict[str, dict[str, int]] = {}
    shared: dict[str, int] = {}

    def typical(path: str) -> float:
        return median(_ratio(data, path, recipe).ratio for recipe in _shared(data, path))

    order = sorted(paths, key=typical)
    for row, path in enumerate(order):
        recipes = _shared(data, path)
        shared[path] = len(recipes)
        counts[path] = {"faster": 0, "indistinguishable": 0, "slower": 0}
        for index, recipe in enumerate(recipes):
            result = _ratio(data, path, recipe)
            jitter = ((index * 7) % 9 - 4) * 0.045
            groups[result.verdict].append((result.ratio, row + jitter))
            counts[path][result.verdict] += 1
    labels = [f"{LABELS[path]} ({shared[path]})" for path in order]
    count_nodes = "\n".join(
        f"\\node[anchor=west,font=\\footnotesize] at (axis cs:7,{row}) {{{_counts(counts[path])}}};"
        for row, path in enumerate(order)
    )
    header = _DOT.join(("faster", "tied", "slower"))
    count_header = (
        f"\\node[anchor=south west,font=\\footnotesize\\bfseries] at (axis cs:7,{len(order) - 0.55}) {{{header}}};"
    )
    slower = _scatter("mark=*, mark size=1.4pt, draw=verdictslower, fill=verdictslower", groups["slower"])
    tied = _scatter("mark=o, mark size=1.6pt, draw=black!60, fill=white", groups["indistinguishable"])
    faster = _scatter("mark=triangle*, mark size=2.2pt, draw=verdictfaster, fill=verdictfaster", groups["faster"])
    return f"""\\begin{{tikzpicture}}
\\begin{{axis}}[width=10.5cm, xmode=log, xmin=0.006, xmax=5,
  xtick={{0.01,0.1,0.25,0.5,1,2,4}}, xticklabels={{0.01,0.1,0.25,0.5,1,2,4}}, {_rows(labels)},
  height={1.4 + 0.75 * len(order):.2f}cm,
  xlabel={{Throughput of the path divided by AX on each shared recipe (log scale)}},
  legend style={{at={{(0.0,1.03)}}, anchor=south west, draw=none, font=\\footnotesize, legend columns=3,
    /tikz/every even column/.append style={{column sep=1.2em}}}}, {_STYLE}]
{count_header}
\\draw[black!50, densely dashed] (axis cs:1,-0.7) -- (axis cs:1,{len(order) - 0.3});
{slower}
\\addlegendentry{{slower than AX}}
{tied}
\\addlegendentry{{tied}}
{faster}
\\addlegendentry{{faster than AX}}
{count_nodes}
\\end{{axis}}
\\end{{tikzpicture}}
"""


def _counts(counts: dict[str, int]) -> str:
    return _DOT.join(str(counts[verdict]) for verdict in ("faster", "indistinguishable", "slower"))


def _shared(data: _PaperData, path: str) -> list[str]:
    return [
        recipe.recipe_id
        for recipe in data.recipes
        if (path, recipe.recipe_id) in data.speed and (_AX, recipe.recipe_id) in data.speed
    ]


def _ratio(data: _PaperData, path: str, recipe: str) -> RatioInterval:
    def speeds(implementation: str) -> list[float]:
        return [record.throughput.value for record in data.groups[implementation, recipe]]

    return ratio_interval(speeds(path), speeds(_AX))


def training_shares(panels: Sequence[tuple[str, Sequence[Row]]], recipe_names: Sequence[str]) -> str:
    """Each path's rate as a share of the model-only ceiling, per recipe, with the seed interval."""
    first = panels[0][1]
    paths = sorted(
        {row.implementation for row in first},
        key=lambda path: median(row.of_ceiling_median for row in first if row.implementation == path),
    )
    labels = [LABELS[path] for path in paths]
    recipes = list(dict.fromkeys(row.recipe_id for row in first))
    offsets = [(index - (len(recipes) - 1) / 2) * 0.18 for index in range(len(recipes))]
    axes = []
    for panel, (title, rows) in enumerate(panels):
        plots = []
        for index, recipe in enumerate(recipes):
            color, mark = _RECIPE_MARKS[index]
            points = " ".join(
                f"({100 * row.of_ceiling_median:.4g},{paths.index(row.implementation) + offsets[index]:.4g}) "
                f"+= ({100 * (row.of_ceiling_high - row.of_ceiling_median):.4g},0) "
                f"-= ({100 * (row.of_ceiling_median - row.of_ceiling_low):.4g},0)"
                for row in rows
                if row.recipe_id == recipe
            )
            legend = f"\\addlegendentry{{{recipe_names[index]}}}" if panel == 0 else ""
            plots.append(
                f"\\addplot[only marks, mark={mark}, mark size=1.7pt, draw={color}, fill={color}, "
                f"error bars/.cd, x dir=both, x explicit, error bar style={{{color}}}] "
                f"coordinates {{{points}}};{legend}"
            )
        position = "" if panel == 0 else ", at={(panel0.east)}, anchor=west, xshift=0.6cm, yticklabels={}"
        legend_style = (
            ", legend style={at={(1.05,-0.22)}, anchor=north, draw=none, font=\\footnotesize, legend columns=4, "
            "/tikz/every even column/.append style={column sep=1.2em}}"
            if panel == 0
            else ""
        )
        axes.append(
            f"\\begin{{axis}}[name=panel{panel}, width=7.2cm, height=7cm, xmin=0, xmax=105, xtick={{0,25,50,75,100}},\n"
            f"  {_rows(labels)}, title={{{title}}}, title style={{font=\\small\\bfseries}},\n"
            f"  xlabel={{Share of the model-only ceiling (\\%)}}{position}{legend_style}, {_STYLE}]\n"
            + "\n".join(plots)
            + "\n\\end{axis}"
        )
    return "\\begin{tikzpicture}\n" + "\n".join(axes) + "\n\\end{tikzpicture}\n"
