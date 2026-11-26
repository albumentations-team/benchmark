# ruff: noqa: INP001
"""PGFPlots sources for the report figures; the paper inputs them and the README renders them to PNG."""

from __future__ import annotations

from math import ceil
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
_STYLE = (
    "axis lines*=left, y axis line style={draw=none}, ytick style={draw=none}, xmajorgrids, "
    "major grid style={black!10}, tick label style={font=\\small}, label style={font=\\small}, clip=false"
)
# A full light frame and a bold title keep side-by-side panels apart.
_PANEL = (
    "axis lines=box, axis line style={black!35}, title style={font=\\small\\bfseries}, "
    "y axis line style={draw=black!35}, xtick pos=bottom"
)
# Color marks the path in every figure (see plot-colors.tex); a recipe is told apart by its marker shape.
_RECIPE_MARKS = ("*", "square*", "diamond*", "triangle*")


def _color(path: str) -> str:
    return "path" + LABELS[path].replace(" ", "").replace("-", "")


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
    dots = "\n".join(
        _scatter(
            f"mark=*, mark size=1.3pt, draw=none, fill={_color(path)}, fill opacity=0.45",
            [(speed, row) for row, speed in recipes if order[row] == path],
        )
        for path in order
    )
    marks = "\n".join(
        _scatter(f"mark=|, mark size=5pt, line width=2pt, draw={_color(path)}!75!black", [(medians[path], row)])
        for row, path in enumerate(order)
    )
    memory_marks = "\n".join(
        _scatter(f"mark=*, mark size=2.2pt, draw={_color(path)}!75!black, fill={_color(path)}", [(memory[path], row)])
        for row, path in enumerate(order)
    )
    speed_max = 1000 * ceil(max(speed for _, speed in recipes) / 1000)
    value_nodes = "\n".join(
        f"\\node[anchor=west,font=\\footnotesize] at (axis cs:{speed_max},{row}) {{\\,{medians[path]:,.0f}}};"
        for row, path in enumerate(order)
    )
    memory_max = 500 * ceil(max(memory.values()) / 500)
    return f"""\\begin{{tikzpicture}}
\\begin{{axis}}[name=speed, width=9.1cm, height=6.6cm, xmin=0, xmax={speed_max},
  xtick distance=1000, {_rows(labels)}, title={{Images per second}},
  xlabel={{One dot per recipe; the bar marks the median}}, {_STYLE}, {_PANEL}]
{dots}
{marks}
{value_nodes}
\\end{{axis}}
\\begin{{axis}}[at={{(speed.east)}}, anchor=west, xshift=1.9cm, width=4.3cm, height=6.6cm, xmin=0,
  xmax={memory_max}, xtick distance=500, ytick=\\empty, ymin=-0.7, ymax={len(order) - 0.3},
  title={{Peak GPU memory}}, xlabel={{MiB, median over recipes}}, {_STYLE}, {_PANEL}]
{memory_marks}
\\end{{axis}}
\\end{{tikzpicture}}
"""


def ax_pairs(data: _PaperData, paths: Sequence[str] | None = None) -> str:
    """Every shared recipe's throughput ratio against AX, marked by what the seed intervals can tell."""
    if paths is None:
        paths = [implementation for implementation, _ in data.implementations if implementation != _AX]
    clear: dict[str, list[tuple[float, float]]] = {}
    tied: dict[str, list[tuple[float, float]]] = {}
    counts: dict[str, dict[str, int]] = {}

    def typical(path: str) -> float:
        return median(_ratio(data, path, recipe).ratio for recipe in _shared(data, path))

    order = sorted(paths, key=typical)
    for row, path in enumerate(order):
        clear[path], tied[path] = [], []
        counts[path] = {"shared": 0, "faster": 0, "indistinguishable": 0, "slower": 0}
        for index, recipe in enumerate(_shared(data, path)):
            result = _ratio(data, path, recipe)
            jitter = ((index * 7) % 9 - 4) * 0.045
            (tied if result.verdict == "indistinguishable" else clear)[path].append((result.ratio, row + jitter))
            counts[path]["shared"] += 1
            counts[path][result.verdict] += 1
    labels = [LABELS[path] for path in order]
    ratio_max = ceil(2 * max(ratio for group in (clear, tied) for points in group.values() for ratio, _ in points)) / 2
    styles = (
        (clear, "mark=*, mark size=1.5pt, draw={0}!75!black, fill={0}"),
        (tied, "mark=o, mark size=1.7pt, thick, draw={0}, fill=white"),
    )
    points = "\n".join(
        _scatter("forget plot, " + style.format(_color(path)), group[path])
        for path in order
        for group, style in styles
        if group[path]
    )
    top = len(order) - 0.3
    table = _count_table(counts, order, ratio_max)
    return f"""\\begin{{tikzpicture}}
\\begin{{axis}}[width=8.2cm, xmin=0, xmax={ratio_max}, xtick distance=1, minor x tick num=1,
  xticklabel={{$\\pgfmathprintnumber{{\\tick}}\\times$}}, {_rows(labels)},
  height={1.4 + 0.75 * len(order):.2f}cm,
  xlabel={{Throughput of the path divided by AX on each shared recipe}},
  legend style={{at={{(0.5,-0.22)}}, anchor=north, draw=none, font=\\footnotesize, legend columns=2,
    /tikz/every even column/.append style={{column sep=1.2em}}}}, {_STYLE}]
\\draw[black!50, densely dashed] (axis cs:1,-0.7) -- (axis cs:1,{top});
\\node[anchor=south east,font=\\footnotesize,black!60] at (axis cs:0.95,{top}) {{$\\leftarrow$ slower than AX}};
\\node[anchor=south west,font=\\footnotesize,black!60] at (axis cs:1.05,{top}) {{faster than AX $\\rightarrow$}};
\\addlegendimage{{only marks, mark=*, mark size=1.5pt, black!70}}
\\addlegendentry{{interval excludes 1}}
\\addlegendimage{{only marks, mark=o, mark size=1.7pt, thick, black!70}}
\\addlegendentry{{tied}}
{points}
{table}
\\end{{axis}}
\\end{{tikzpicture}}
"""


# Right edge of each table column, in points past the plot's right edge.
_TABLE_COLUMNS = (
    ("shared", "shared", 40),
    ("faster", "faster", 80),
    ("tied", "indistinguishable", 112),
    ("slower", "slower", 152),
)


def _count_table(counts: dict[str, dict[str, int]], order: Sequence[str], ratio_max: float) -> str:
    """A small table to the right of the plot: per path, shared recipes and the faster, tied, and slower counts."""
    header_y = len(order) - 0.45
    lines = [
        f"\\node[anchor=south east,xshift={edge}pt,font=\\footnotesize\\bfseries] at (axis cs:{ratio_max},{header_y}) "
        f"{{{title}}};"
        for title, _, edge in _TABLE_COLUMNS
    ]
    lines.append(
        f"\\draw[black!40] ([xshift=4pt]axis cs:{ratio_max},{header_y}) -- "
        f"([xshift={_TABLE_COLUMNS[-1][2]}pt]axis cs:{ratio_max},{header_y});"
    )
    lines.extend(
        f"\\node[anchor=east,xshift={edge}pt,font=\\footnotesize] at (axis cs:{ratio_max},{row}) "
        f"{{{counts[path][key]}}};"
        for row, path in enumerate(order)
        for _, key, edge in _TABLE_COLUMNS
    )
    return "\n".join(lines)


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
        if panel == 0:
            plots.extend(
                f"\\addlegendimage{{only marks, mark={mark}, mark size=1.7pt, black!70}}"
                f"\\addlegendentry{{{recipe_names[index]}}}"
                for index, mark in enumerate(_RECIPE_MARKS[: len(recipes)])
            )
        for index, recipe in enumerate(recipes):
            mark = _RECIPE_MARKS[index]
            for row in rows:
                if row.recipe_id != recipe:
                    continue
                color = _color(row.implementation)
                plots.append(
                    f"\\addplot[forget plot, only marks, mark={mark}, mark size=1.7pt, draw={color}!75!black, "
                    f"fill={color}, error bars/.cd, x dir=both, x explicit, error bar style={{{color}}}] coordinates "
                    f"{{({100 * row.of_ceiling_median:.4g},{paths.index(row.implementation) + offsets[index]:.4g}) "
                    f"+= ({100 * (row.of_ceiling_high - row.of_ceiling_median):.4g},0) "
                    f"-= ({100 * (row.of_ceiling_median - row.of_ceiling_low):.4g},0)}};"
                )
        position = "" if panel == 0 else ", at={(panel0.east)}, anchor=west, xshift=0.9cm, yticklabels={}"
        legend_style = (
            ", legend style={at={(1.05,-0.22)}, anchor=north, draw=none, font=\\footnotesize, legend columns=4, "
            "/tikz/every even column/.append style={column sep=1.2em}}"
            if panel == 0
            else ""
        )
        axes.append(
            f"\\begin{{axis}}[name=panel{panel}, width=7.2cm, height=7cm, xmin=0, xmax=105, xtick={{0,25,50,75,100}},\n"
            f"  {_rows(labels)}, title={{{title}}},\n"
            f"  xlabel={{Share of the model-only ceiling (\\%)}}{position}{legend_style}, {_STYLE}, {_PANEL}]\n"
            + "\n".join(plots)
            + "\n\\end{axis}"
        )
    return "\\begin{tikzpicture}\n" + "\n".join(axes) + "\n\\end{tikzpicture}\n"
