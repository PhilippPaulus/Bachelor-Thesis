from __future__ import annotations

import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


def generate_all_plots(run_dir: str | Path) -> list[Path]:
    root = Path(run_dir).expanduser().resolve()
    output = root / "plots"
    output.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    generated.extend(_experiment_1(root, output))
    generated.extend(_experiment_2(root, output))
    generated.extend(_experiment_3(root, output))
    generated.extend(_experiment_4(root, output))
    generated.extend(_experiment_5(root, output))
    return generated


def _experiment_1(root: Path, output: Path) -> list[Path]:
    single = _read(root / "experiment_1_accuracy" / "single_table_results.csv")
    complete = _read(root / "experiment_1_accuracy" / "complete_query_results.csv")
    generated: list[Path] = []
    valid_single = [row for row in single if row.get("status") == "ok"]
    valid_complete = [row for row in complete if row.get("status") == "ok"]
    if valid_single:
        generated.append(_ecdf(
            output / "experiment_1_single_table_q_error_ecdf.png",
            {
                "Native": [_float(row, "native_q_error") for row in valid_single],
                "Learned": [_float(row, "learned_q_error") for row in valid_single],
            },
            "Single-table Q-error ECDF",
        ))
        generated.append(_scatter(
            output / "experiment_1_learned_vs_native_q_error.png",
            [_float(row, "native_q_error") for row in valid_single],
            [_float(row, "learned_q_error") for row in valid_single],
            "Native Q-error",
            "Learned Q-error",
            "Single-table Q-error comparison",
            log=True,
            diagonal=True,
        ))
    if valid_complete:
        generated.append(_scatter(
            output / "experiment_1_final_query_q_error.png",
            [_float(row, "native_q_error") for row in valid_complete],
            [_float(row, "learned_base_q_error") for row in valid_complete],
            "Native final Q-error",
            "Learned-base final Q-error",
            "Complete-query estimate propagation",
            log=True,
            diagonal=True,
        ))
        generated.append(_scatter(
            output / "experiment_1_exact_base_attainable_improvement.png",
            [_float(row, "learned_improvement_ratio") for row in valid_complete],
            [_float(row, "exact_base_attainable_improvement_ratio") for row in valid_complete],
            "Learned improvement ratio",
            "Exact-base attainable improvement ratio",
            "Learned versus attainable improvement",
            log=True,
            diagonal=True,
        ))
    return generated


def _experiment_2(root: Path, output: Path) -> list[Path]:
    groups = _read(root / "experiment_2_failures" / "grouped_failures.csv")
    worst = _read(root / "experiment_2_failures" / "worst_queries.csv")
    generated: list[Path] = []
    for dimension, filename, title in (
        ("predicate_count", "experiment_2_q_error_by_predicate_count.png", "Q-error by predicate count"),
        ("selectivity_bucket", "experiment_2_q_error_by_selectivity.png", "Q-error by selectivity bucket"),
    ):
        selected = [row for row in groups if row.get("group_dimension") == dimension]
        if selected:
            generated.append(_bar(
                output / filename,
                [row["group_value"] for row in selected],
                [_float(row, "learned_median_q_error") for row in selected],
                title,
                "Median learned Q-error",
                log=True,
            ))
    table_column = [row for row in groups if row.get("group_dimension") in {"table", "constrained_column"}]
    if table_column:
        selected = sorted(table_column, key=lambda row: _float(row, "learned_median_q_error"), reverse=True)[:15]
        generated.append(_bar(
            output / "experiment_2_worst_tables_columns.png",
            [f"{row['group_dimension']}:{row['group_value']}" for row in selected],
            [_float(row, "learned_median_q_error") for row in selected],
            "Worst tables and constrained columns",
            "Median learned Q-error",
            log=True,
        ))
    if worst:
        generated.append(_hist(
            output / "experiment_2_over_under_distribution.png",
            [math.log10(max(_float(row, "learned_signed_error_ratio"), 1e-12)) for row in worst],
            "Learned over/underestimation among worst queries",
            "log10(estimate / exact)",
        ))
    return generated


def _experiment_3(root: Path, output: Path) -> list[Path]:
    rows = _read(root / "experiment_3_base_join_influence" / "base_join_changes.csv")
    valid = [row for row in rows if _bool(row.get("applicable"))]
    if not valid:
        return []
    generated = [_bar(
        output / "experiment_3_changed_vs_unchanged.png",
        ["Changed", "Unchanged"],
        [sum(_bool(row["base_join_changed_native_learned"]) for row in valid), sum(not _bool(row["base_join_changed_native_learned"]) for row in valid)],
        "Learned-base influence on first base join",
        "Query count",
    )]
    by_size: dict[str, list[bool]] = defaultdict(list)
    for row in valid:
        by_size[row["relation_count"]].append(_bool(row["base_join_changed_native_learned"]))
    generated.append(_bar(
        output / "experiment_3_change_rate_by_query_size.png",
        sorted(by_size, key=int),
        [sum(by_size[key]) / len(by_size[key]) for key in sorted(by_size, key=int)],
        "Base-join change rate by query size",
        "Change rate",
    ))
    generated.append(_scatter(
        output / "experiment_3_change_probability_vs_divergence.png",
        [_float(row, "maximum_absolute_log_cardinality_difference") for row in valid],
        [1.0 if _bool(row["base_join_changed_native_learned"]) else 0.0 for row in valid],
        "Maximum absolute log estimate difference",
        "Base join changed (0/1)",
        "Base-join change versus estimate divergence",
    ))
    return generated


def _experiment_4(root: Path, output: Path) -> list[Path]:
    rows = _read(root / "experiment_4_base_join_quality" / "decision_quality.csv")
    valid = [row for row in rows if row.get("status") == "ok"]
    if not valid:
        return []
    counts = Counter(row["decision_category"] for row in valid)
    generated = [_bar(
        output / "experiment_4_agreement_categories.png",
        ["improved", "degraded", "both agree", "neither agrees"],
        [counts[key] for key in ("improved", "degraded", "both agree", "neither agrees")],
        "Agreement with exact-base reference",
        "Query count",
    )]
    generated.append(_scatter(
        output / "experiment_4_relative_first_join_output.png",
        [_float(row, "native_relative_first_join_output") for row in valid],
        [_float(row, "learned_relative_first_join_output") for row in valid],
        "Native relative first-join output",
        "Learned relative first-join output",
        "Exact first-join output relative to exact-base reference",
        log=True,
        diagonal=True,
    ))
    return generated


def _experiment_5(root: Path, output: Path) -> list[Path]:
    rows = _read(root / "experiment_5_runtime" / "runtime_per_query.csv")
    valid = [row for row in rows if row.get("status") == "complete"]
    if not valid:
        return []
    generated = [_scatter(
        output / "experiment_5_runtime_scatter.png",
        [_float(row, "native_median_execution_time_ms") for row in valid],
        [_float(row, "learned_median_execution_time_ms") for row in valid],
        "Native median execution time (ms)",
        "Learned median execution time (ms)",
        "Per-query runtime",
        log=True,
        diagonal=True,
    )]
    generated.append(_hist(
        output / "experiment_5_speedup_distribution.png",
        [_float(row, "speedup_native_over_learned") for row in valid],
        "Runtime speedup distribution",
        "Speedup (native / learned)",
    ))
    changed = [_float(row, "speedup_native_over_learned") for row in valid if _bool(row["full_plan_changed"])]
    unchanged = [_float(row, "speedup_native_over_learned") for row in valid if not _bool(row["full_plan_changed"])]
    generated.append(_box(
        output / "experiment_5_speedup_changed_unchanged.png",
        [changed, unchanged],
        ["Plan changed", "Plan unchanged"],
        "Speedup for changed versus unchanged plans",
    ))
    native = sorted(_float(row, "native_median_execution_time_ms") for row in valid)
    learned = sorted(_float(row, "learned_median_execution_time_ms") for row in valid)
    generated.append(_lines(
        output / "experiment_5_cumulative_workload_runtime.png",
        {
            "Native": _cumulative(native),
            "Learned": _cumulative(learned),
        },
        "Cumulative workload runtime",
        "Queries ordered by runtime",
        "Cumulative execution time (ms)",
    ))
    return generated


def _plot(path: Path, draw: Callable[[Any], None]) -> Path:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        ax = _PillowAxes()
        draw(ax)
        ax.save(path)
        return path
    fig, ax = plt.subplots(figsize=(8, 5))
    draw(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _scatter(path: Path, x: list[float], y: list[float], xlabel: str, ylabel: str, title: str, *, log: bool = False, diagonal: bool = False) -> Path:
    def draw(ax: Any) -> None:
        ax.scatter(x, y, alpha=0.55, s=20)
        if log:
            ax.set_xscale("log")
            ax.set_yscale("log")
        if diagonal and x and y:
            lower, upper = min([*x, *y]), max([*x, *y])
            ax.plot([lower, upper], [lower, upper], linestyle="--", color="black", linewidth=1)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        ax.grid(alpha=0.25)
    return _plot(path, draw)


def _ecdf(path: Path, series: dict[str, list[float]], title: str) -> Path:
    def draw(ax: Any) -> None:
        for label, values in series.items():
            ordered = sorted(values)
            ax.step(ordered, [(index + 1) / len(ordered) for index in range(len(ordered))], where="post", label=label)
        ax.set_xscale("log")
        ax.set(xlabel="Q-error", ylabel="Cumulative fraction", title=title)
        ax.legend()
        ax.grid(alpha=0.25)
    return _plot(path, draw)


def _bar(path: Path, labels: list[str], values: list[float], title: str, ylabel: str, *, log: bool = False) -> Path:
    def draw(ax: Any) -> None:
        ax.bar(range(len(values)), values)
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        if log:
            ax.set_yscale("log")
        ax.set(ylabel=ylabel, title=title)
        ax.grid(axis="y", alpha=0.25)
    return _plot(path, draw)


def _hist(path: Path, values: list[float], title: str, xlabel: str) -> Path:
    return _plot(path, lambda ax: (ax.hist(values, bins=min(30, max(5, len(values) // 3))), ax.set(xlabel=xlabel, ylabel="Count", title=title)))


def _box(path: Path, values: list[list[float]], labels: list[str], title: str) -> Path:
    populated = [(value, label) for value, label in zip(values, labels) if value]
    cleaned = [value for value, _ in populated]
    cleaned_labels = [label for _, label in populated]
    return _plot(path, lambda ax: (ax.boxplot(cleaned, tick_labels=cleaned_labels), ax.set(ylabel="Speedup (native / learned)", title=title)))


def _lines(path: Path, series: dict[str, list[float]], title: str, xlabel: str, ylabel: str) -> Path:
    def draw(ax: Any) -> None:
        for label, values in series.items():
            ax.plot(range(1, len(values) + 1), values, label=label)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        ax.legend()
        ax.grid(alpha=0.25)
    return _plot(path, draw)


def _read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def _bool(value: Any) -> bool:
    return value is True or str(value).lower() == "true"


def _cumulative(values: list[float]) -> list[float]:
    total = 0.0
    output: list[float] = []
    for value in values:
        total += value
        output.append(total)
    return output


class _PillowAxes:
    """Small fallback renderer for environments without Matplotlib."""

    _COLORS = ("#2f6f9f", "#d76b35", "#4f9d69", "#8b5fbf", "#b79a2f")

    def __init__(self) -> None:
        self.commands: list[dict[str, Any]] = []
        self.xlabel = ""
        self.ylabel = ""
        self.title = ""
        self.xscale = "linear"
        self.yscale = "linear"
        self.tick_labels: list[str] = []

    def scatter(self, x: list[float], y: list[float], **kwargs: Any) -> None:
        self.commands.append({"kind": "scatter", "x": list(x), "y": list(y), "label": kwargs.get("label")})

    def step(self, x: list[float], y: list[float], **kwargs: Any) -> None:
        self.commands.append({"kind": "line", "x": list(x), "y": list(y), "label": kwargs.get("label")})

    def plot(self, x: list[float] | range, y: list[float], **kwargs: Any) -> None:
        self.commands.append({
            "kind": "line", "x": list(x), "y": list(y), "label": kwargs.get("label"),
            "color": kwargs.get("color"), "dashed": kwargs.get("linestyle") == "--",
        })

    def bar(self, x: list[float] | range, values: list[float], **_: Any) -> None:
        self.commands.append({"kind": "bar", "x": list(x), "y": list(values)})

    def hist(self, values: list[float], bins: int = 10, **_: Any) -> None:
        if not values:
            return
        low, high = min(values), max(values)
        if low == high:
            low -= 0.5
            high += 0.5
        width = (high - low) / bins
        counts = [0] * bins
        for value in values:
            index = min(int((value - low) / width), bins - 1)
            counts[index] += 1
        centers = [low + (index + 0.5) * width for index in range(bins)]
        self.commands.append({"kind": "bar", "x": centers, "y": counts, "bar_width": width * 0.9})

    def boxplot(self, values: list[list[float]], *, tick_labels: list[str], **_: Any) -> None:
        self.tick_labels = list(tick_labels)
        self.commands.append({"kind": "box", "x": list(range(1, len(values) + 1)), "values": values})

    def set_xscale(self, scale: str) -> None:
        self.xscale = scale

    def set_yscale(self, scale: str) -> None:
        self.yscale = scale

    def set(self, **kwargs: Any) -> None:
        self.xlabel = str(kwargs.get("xlabel", self.xlabel))
        self.ylabel = str(kwargs.get("ylabel", self.ylabel))
        self.title = str(kwargs.get("title", self.title))

    def set_xticks(self, _ticks: Any, labels: list[str], **_: Any) -> None:
        self.tick_labels = [str(label) for label in labels]

    def legend(self, **_: Any) -> None:
        return None

    def grid(self, *_: Any, **__: Any) -> None:
        return None

    def save(self, path: Path) -> None:
        from PIL import Image, ImageDraw, ImageFont

        width, height = 1280, 800
        left, right, top, bottom = 125, 45, 85, 145
        plot_left, plot_right = left, width - right
        plot_top, plot_bottom = top, height - bottom
        image = Image.new("RGB", (width, height), "white")
        canvas = ImageDraw.Draw(image)
        font = ImageFont.load_default(size=18)
        small = ImageFont.load_default(size=14)
        title_font = ImageFont.load_default(size=24)

        x_values, y_values = self._data_bounds()
        transformed_x = [self._transform(value, self.xscale) for value in x_values]
        transformed_y = [self._transform(value, self.yscale) for value in y_values]
        x_min, x_max = _range(transformed_x)
        y_min, y_max = _range(transformed_y, include_zero=self.yscale != "log")

        def px(value: float) -> float:
            transformed = self._transform(value, self.xscale)
            return plot_left + (transformed - x_min) / (x_max - x_min) * (plot_right - plot_left)

        def py(value: float) -> float:
            transformed = self._transform(value, self.yscale)
            return plot_bottom - (transformed - y_min) / (y_max - y_min) * (plot_bottom - plot_top)

        canvas.line((plot_left, plot_top, plot_left, plot_bottom), fill="black", width=2)
        canvas.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="black", width=2)
        for index in range(6):
            y = plot_top + index * (plot_bottom - plot_top) / 5
            canvas.line((plot_left, y, plot_right, y), fill="#dddddd", width=1)
        legend: list[tuple[str, str]] = []
        for index, command in enumerate(self.commands):
            color = command.get("color") or self._COLORS[index % len(self._COLORS)]
            if command.get("label"):
                legend.append((str(command["label"]), color))
            if command["kind"] == "scatter":
                for x, y in zip(command["x"], command["y"]):
                    cx, cy = px(float(x)), py(float(y))
                    canvas.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=color)
            elif command["kind"] == "line":
                points = [(px(float(x)), py(float(y))) for x, y in zip(command["x"], command["y"])]
                if command.get("dashed"):
                    _draw_dashed(canvas, points, color)
                elif len(points) >= 2:
                    canvas.line(points, fill=color, width=3)
            elif command["kind"] == "bar":
                xs = [float(value) for value in command["x"]]
                auto_width = (x_max - x_min) / max(len(xs), 1) * 0.7
                bar_width = float(command.get("bar_width", auto_width))
                baseline = 10**y_min if self.yscale == "log" else 0.0
                for x, y in zip(xs, command["y"]):
                    x1, x2 = px(x - bar_width / 2), px(x + bar_width / 2)
                    canvas.rectangle((x1, py(float(y)), x2, py(baseline)), fill=color, outline="white")
            elif command["kind"] == "box":
                for x, values in zip(command["x"], command["values"]):
                    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
                    if not clean:
                        continue
                    q1, median, q3 = _quantile(clean, 0.25), _quantile(clean, 0.5), _quantile(clean, 0.75)
                    x1, x2, center = px(x - 0.25), px(x + 0.25), px(x)
                    canvas.line((center, py(clean[0]), center, py(clean[-1])), fill=color, width=2)
                    canvas.rectangle((x1, py(q3), x2, py(q1)), outline=color, width=3)
                    canvas.line((x1, py(median), x2, py(median)), fill=color, width=3)

        canvas.text((width / 2, 30), self.title, fill="black", font=title_font, anchor="mm")
        canvas.text((width / 2, height - 35), self.xlabel, fill="black", font=font, anchor="mm")
        canvas.text((12, 55), self.ylabel, fill="black", font=font)
        canvas.text((plot_left, plot_bottom + 12), _display_scale_value(x_min, self.xscale), fill="black", font=small)
        canvas.text((plot_right, plot_bottom + 12), _display_scale_value(x_max, self.xscale), fill="black", font=small, anchor="ra")
        canvas.text((plot_left - 10, plot_bottom), _display_scale_value(y_min, self.yscale), fill="black", font=small, anchor="ra")
        canvas.text((plot_left - 10, plot_top), _display_scale_value(y_max, self.yscale), fill="black", font=small, anchor="ra")
        if self.tick_labels:
            positions = range(len(self.tick_labels))
            if any(command["kind"] == "box" for command in self.commands):
                positions = range(1, len(self.tick_labels) + 1)
            for position, label in zip(positions, self.tick_labels):
                canvas.text((px(float(position)), plot_bottom + 30), label[:24], fill="black", font=small, anchor="ma")
        for index, (label, color) in enumerate(legend):
            y = plot_top + 10 + index * 24
            canvas.rectangle((plot_right - 180, y, plot_right - 164, y + 12), fill=color)
            canvas.text((plot_right - 155, y - 3), label, fill="black", font=small)
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path, format="PNG")

    def _data_bounds(self) -> tuple[list[float], list[float]]:
        xs: list[float] = []
        ys: list[float] = []
        for command in self.commands:
            if command["kind"] in {"scatter", "line", "bar"}:
                xs.extend(float(value) for value in command["x"])
                ys.extend(float(value) for value in command["y"])
            elif command["kind"] == "box":
                xs.extend(float(value) for value in command["x"])
                ys.extend(
                    float(value)
                    for values in command["values"]
                    for value in values
                    if math.isfinite(float(value))
                )
        return (xs or [0.0, 1.0], ys or [0.0, 1.0])

    @staticmethod
    def _transform(value: float, scale: str) -> float:
        return math.log10(max(float(value), 1e-12)) if scale == "log" else float(value)


def _range(values: list[float], *, include_zero: bool = False) -> tuple[float, float]:
    low, high = min(values), max(values)
    if include_zero:
        low = min(low, 0.0)
        high = max(high, 0.0)
    if low == high:
        margin = max(abs(low) * 0.1, 1.0)
        low -= margin
        high += margin
    else:
        margin = (high - low) * 0.05
        low -= margin
        high += margin
    return low, high


def _quantile(values: list[float], quantile: float) -> float:
    position = (len(values) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    return values[lower] if lower == upper else values[lower] + (values[upper] - values[lower]) * (position - lower)


def _draw_dashed(canvas: Any, points: list[tuple[float, float]], color: str) -> None:
    if len(points) < 2:
        return
    for start, end in zip(points, points[1:]):
        x1, y1 = start
        x2, y2 = end
        distance = math.hypot(x2 - x1, y2 - y1)
        segments = max(int(distance / 10), 1)
        for index in range(0, segments, 2):
            a, b = index / segments, min((index + 1) / segments, 1.0)
            canvas.line((x1 + (x2 - x1) * a, y1 + (y2 - y1) * a, x1 + (x2 - x1) * b, y1 + (y2 - y1) * b), fill=color, width=2)


def _display_scale_value(value: float, scale: str) -> str:
    actual = 10**value if scale == "log" else value
    return f"{actual:.3g}"
