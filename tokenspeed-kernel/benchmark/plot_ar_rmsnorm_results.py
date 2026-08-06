#!/usr/bin/env python3
"""Plot tracked AR+RMSNorm graph and end-to-end result summaries.

The plotting inputs are compact JSON files emitted by the TokenSpeed benchmark
analysis pipeline, not the raw campaign trees:

* ``graph-sweep-summary.json`` for captured graph comparisons;
* ``end-to-end-summary.json`` for restart-block serving comparisons.

Matplotlib is imported only when rendering so schema extraction and tests do
not require the optional plotting dependency.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

DEFAULT_CONTRAST = "triton_profile_vs_upstream_unfused"
GRAPH_FIELDS = {
    "world_size",
    "N",
    "calls_per_graph",
    "M",
    "triton_vs_unfused_pct",
    "triton_vs_unfused_adjusted_pct",
    "triton_vs_iris_pct",
    "unfused_path",
}
METRIC_LABELS = {
    "output_throughput": "Output throughput change (%)",
    "median_tpot_ms": "Median TPOT change (%)",
}
COLORS = {
    "unfused": "tab:blue",
    "adjusted": "tab:orange",
    "iris": "tab:green",
    2: "tab:gray",
    4: "tab:blue",
    8: "tab:orange",
}
MARKERS = ("o", "s", "^", "D")


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object with a useful top-level validation error."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def select_graph_comparisons(
    summary: dict[str, Any],
    *,
    world_sizes: list[int],
    calls_per_graph: int,
    min_m: int | None = None,
    max_m: int | None = None,
) -> dict[int, list[dict[str, Any]]]:
    """Select complete graph-comparison rows without mixing graph site counts."""
    if summary.get("schema_version") != 2:
        raise ValueError(
            "graph summary must use schema_version=2 with comparison rows"
        )
    comparisons = summary.get("comparisons")
    if not isinstance(comparisons, list):
        raise ValueError("graph summary is missing comparisons[]")

    selected: dict[int, list[dict[str, Any]]] = {}
    for ws in world_sizes:
        rows = []
        for row in comparisons:
            if not isinstance(row, dict):
                raise ValueError("graph comparisons[] must contain objects")
            if int(row.get("world_size", -1)) != ws:
                continue
            if int(row.get("calls_per_graph", -1)) != calls_per_graph:
                continue
            missing = GRAPH_FIELDS - row.keys()
            if missing:
                raise ValueError(
                    f"WS={ws} graph comparison is missing {sorted(missing)}"
                )
            m = int(row["M"])
            if min_m is not None and m < min_m:
                continue
            if max_m is not None and m > max_m:
                continue
            rows.append(row)
        rows.sort(key=lambda row: int(row["M"]))
        if not rows:
            raise ValueError(
                f"no graph rows for WS={ws}, calls_per_graph={calls_per_graph}, "
                f"M range {min_m or '-inf'}..{max_m or '+inf'}"
            )
        selected[ws] = rows
    return selected


def extract_e2e_metrics(
    summary: dict[str, Any],
    *,
    world_sizes: list[int],
    contrast: str = DEFAULT_CONTRAST,
) -> dict[int, dict[str, dict[str, float | int]]]:
    """Extract aggregate throughput and TPOT statistics for each world size."""
    if summary.get("schema_version") != 1:
        raise ValueError("end-to-end summary must use schema_version=1")
    all_world_sizes = summary.get("world_sizes")
    if not isinstance(all_world_sizes, dict):
        raise ValueError("end-to-end summary is missing world_sizes{}")

    result = {}
    for ws in world_sizes:
        try:
            statistics_payload = all_world_sizes[str(ws)]["contrasts"][contrast][
                "statistics"
            ]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"end-to-end summary has no {contrast!r} statistics for WS={ws}"
            ) from exc
        metrics = {}
        for metric in METRIC_LABELS:
            values = statistics_payload.get(metric)
            required = {"mean", "ci95_low", "ci95_high", "n_blocks", "n_pairs"}
            if not isinstance(values, dict) or not required <= values.keys():
                raise ValueError(
                    f"WS={ws} {contrast} is missing complete {metric} statistics"
                )
            metrics[metric] = {
                "mean": float(values["mean"]),
                "ci95_low": float(values["ci95_low"]),
                "ci95_high": float(values["ci95_high"]),
                "n_blocks": int(values["n_blocks"]),
                "n_pairs": int(values["n_pairs"]),
            }
        result[ws] = metrics
    return result


def extract_e2e_blocks(
    summary: dict[str, Any],
    *,
    world_size: int,
    contrast: str = DEFAULT_CONTRAST,
) -> dict[str, list[tuple[int, float]]]:
    """Extract block-level means for throughput and TPOT paired changes."""
    try:
        paired = summary["world_sizes"][str(world_size)]["contrasts"][contrast][
            "paired_values_pct"
        ]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"end-to-end summary has no {contrast!r} paired values for "
            f"WS={world_size}"
        ) from exc

    result = {}
    for metric in METRIC_LABELS:
        entries = paired.get(metric)
        if not isinstance(entries, list) or not entries:
            raise ValueError(
                f"WS={world_size} {contrast} has no paired {metric} values"
            )
        blocks = []
        for entry in entries:
            values = entry.get("values") if isinstance(entry, dict) else None
            if not isinstance(values, list) or not values:
                raise ValueError(
                    f"WS={world_size} {metric} block entry has no values"
                )
            blocks.append(
                (
                    int(entry["block"]),
                    statistics.fmean(float(value) for value in values),
                )
            )
        result[metric] = sorted(blocks)
    return result


def _load_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError as exc:
        raise RuntimeError(
            "plotting AR+RMSNorm results requires matplotlib; "
            "install it in the benchmark environment"
        ) from exc
    return plt, mticker


def _format_m_axis(ax, rows: list[dict[str, Any]]) -> None:
    values = [int(row["M"]) for row in rows]
    ax.set_xscale("log", base=2)
    preferred = (1, 2, 4, 8, 16, 24, 32, 40, 48, 64, 128, 256)
    ticks = [value for value in preferred if min(values) <= value <= max(values)]
    ax.set_xticks(ticks, labels=[str(value) for value in ticks])
    ax.tick_params(axis="x", labelrotation=0)


def _finish_figure(fig, output: Path, *, caption: str) -> None:
    if caption:
        fig.text(
            0.5,
            0.015,
            caption,
            ha="center",
            va="bottom",
            fontsize=8,
            style="italic",
            alpha=0.8,
        )
    fig.tight_layout(rect=(0, 0.055 if caption else 0, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    fig.clear()
    print(f"wrote {output}")


def plot_graph_overview(
    selected: dict[int, list[dict[str, Any]]],
    *,
    output: Path,
    title: str,
    caption: str,
) -> None:
    """Plot candidate deltas versus upstream-unfused and Iris by world size."""
    plt, _ = _load_pyplot()
    world_sizes = list(selected)
    fig, axes = plt.subplots(
        1,
        len(world_sizes),
        figsize=(6.2 * len(world_sizes), 4.8),
        sharey=True,
    )
    if len(world_sizes) == 1:
        axes = [axes]

    for ax, ws in zip(axes, world_sizes):
        rows = selected[ws]
        m_values = [int(row["M"]) for row in rows]
        ax.plot(
            m_values,
            [float(row["triton_vs_unfused_pct"]) for row in rows],
            color=COLORS["unfused"],
            marker=MARKERS[0],
            linewidth=1.8,
            label="Triton vs upstream-unfused (raw)",
        )
        ax.plot(
            m_values,
            [float(row["triton_vs_iris_pct"]) for row in rows],
            color=COLORS["iris"],
            marker=MARKERS[1],
            linewidth=1.8,
            label="Triton vs Iris fused",
        )
        ax.axhline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.65)
        if min(m_values) <= 43 <= max(m_values):
            ax.axvline(43, color="black", linewidth=0.9, linestyle=":", alpha=0.65)
            ax.text(
                43,
                0.98,
                " M43 RCCL switch",
                ha="left",
                va="top",
                fontsize=8,
                transform=ax.get_xaxis_transform(),
            )
        _format_m_axis(ax, rows)
        ax.grid(True, which="major", linestyle=":", alpha=0.45)
        ax.set_xlabel("M (rows)")
        ax.set_title(f"World size {ws}")
        if ax is axes[0]:
            ax.set_ylabel("Candidate latency change (%)\nnegative is faster")

    axes[0].legend(loc="best", fontsize=8.5)
    fig.suptitle(title, fontsize=13)
    _finish_figure(fig, output, caption=caption)
    plt.close(fig)


def plot_graph_border(
    rows: list[dict[str, Any]],
    *,
    output: Path,
    title: str,
    caption: str,
) -> None:
    """Plot raw and reset-copy-adjusted candidate deltas around one border."""
    plt, mticker = _load_pyplot()
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    m_values = [int(row["M"]) for row in rows]
    ax.plot(
        m_values,
        [float(row["triton_vs_unfused_pct"]) for row in rows],
        color=COLORS["unfused"],
        marker=MARKERS[0],
        linewidth=2,
        label="Raw replay",
    )
    ax.plot(
        m_values,
        [float(row["triton_vs_unfused_adjusted_pct"]) for row in rows],
        color=COLORS["adjusted"],
        marker=MARKERS[1],
        linewidth=2,
        label="Reset-copy-adjusted",
    )
    ax.axhline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.7)
    ax.axvspan(min(m_values), 40.5, color=COLORS["unfused"], alpha=0.08)
    ax.axvline(42.5, color="black", linewidth=1, linestyle=":", alpha=0.7)
    ax.text(
        42.55,
        ax.get_ylim()[1],
        "M43: upstream switches to RCCL",
        ha="left",
        va="top",
        fontsize=8.5,
    )
    for target_m in (40, 41, 42, 43):
        row = next((row for row in rows if int(row["M"]) == target_m), None)
        if row is None:
            continue
        value = float(row["triton_vs_unfused_adjusted_pct"])
        ax.annotate(
            f"{value:+.2f}%",
            (target_m, value),
            xytext=(0, 9 if value <= 0 else -14),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    ax.set_xticks(m_values)
    ax.xaxis.set_major_locator(mticker.FixedLocator(m_values))
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.set_xlabel("M (rows)")
    ax.set_ylabel("Triton latency change vs upstream-unfused (%)\nnegative is faster")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)
    _finish_figure(fig, output, caption=caption)
    plt.close(fig)


def plot_e2e_summary(
    metrics: dict[int, dict[str, dict[str, float | int]]],
    *,
    output: Path,
    title: str,
    caption: str,
) -> None:
    """Plot aggregate serving changes and confidence intervals by world size."""
    plt, _ = _load_pyplot()
    world_sizes = list(metrics)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for ax, metric in zip(axes, METRIC_LABELS):
        means = [float(metrics[ws][metric]["mean"]) for ws in world_sizes]
        lower = [
            mean - float(metrics[ws][metric]["ci95_low"])
            for ws, mean in zip(world_sizes, means)
        ]
        upper = [
            float(metrics[ws][metric]["ci95_high"]) - mean
            for ws, mean in zip(world_sizes, means)
        ]
        bars = ax.bar(
            [str(ws) for ws in world_sizes],
            means,
            color=[COLORS.get(ws, "tab:gray") for ws in world_sizes],
            yerr=[lower, upper],
            capsize=5,
            alpha=0.9,
        )
        ax.axhline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.7)
        ax.grid(True, axis="y", linestyle=":", alpha=0.45)
        ax.set_xlabel("World size")
        ax.set_ylabel(METRIC_LABELS[metric])
        direction = "higher is better" if metric == "output_throughput" else "lower is better"
        ax.set_title(direction)
        for bar, mean in zip(bars, means):
            ax.annotate(
                f"{mean:+.2f}%",
                (bar.get_x() + bar.get_width() / 2, mean),
                xytext=(0, 5 if mean >= 0 else -13),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
            )
    fig.suptitle(title, fontsize=13)
    _finish_figure(fig, output, caption=caption)
    plt.close(fig)


def plot_e2e_blocks(
    blocks: dict[str, list[tuple[int, float]]],
    *,
    world_size: int,
    output: Path,
    title: str,
    caption: str,
) -> None:
    """Plot block-level paired serving changes for one world size."""
    plt, _ = _load_pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for ax, metric in zip(axes, METRIC_LABELS):
        entries = blocks[metric]
        labels = [str(block) for block, _ in entries]
        values = [value for _, value in entries]
        color = COLORS["unfused"] if metric == "output_throughput" else COLORS["adjusted"]
        bars = ax.bar(labels, values, color=color, alpha=0.9)
        ax.axhline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.7)
        ax.grid(True, axis="y", linestyle=":", alpha=0.45)
        ax.set_xlabel("Fresh-server restart block")
        ax.set_ylabel(METRIC_LABELS[metric])
        direction = "higher is better" if metric == "output_throughput" else "lower is better"
        ax.set_title(direction)
        for bar, value in zip(bars, values):
            ax.annotate(
                f"{value:+.2f}%",
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 4 if value >= 0 else -13),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
    fig.suptitle(f"{title} (WS{world_size})", fontsize=13)
    _finish_figure(fig, output, caption=caption)
    plt.close(fig)


def _add_common_output_args(parser: argparse.ArgumentParser, *, title: str) -> None:
    parser.add_argument("input", type=Path, help="Tracked summary JSON")
    parser.add_argument("output", type=Path, help="Output PNG path")
    parser.add_argument("--title", default=title)
    parser.add_argument("--caption", default="")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    overview = subparsers.add_parser(
        "graph-overview",
        help="Plot graph deltas versus upstream-unfused and Iris",
    )
    _add_common_output_args(
        overview,
        title="GLM-5.2-FP8 captured AR+RMSNorm comparison",
    )
    overview.add_argument("--world-sizes", type=int, nargs="+", default=[4, 8])
    overview.add_argument("--calls-per-graph", type=int, default=156)
    overview.add_argument("--min-m", type=int)
    overview.add_argument("--max-m", type=int)

    border = subparsers.add_parser(
        "graph-border",
        help="Plot raw and adjusted graph deltas around a profitability border",
    )
    _add_common_output_args(
        border,
        title="GLM-5.2-FP8 WS8 profitability border",
    )
    border.add_argument("--world-size", type=int, default=8)
    border.add_argument("--calls-per-graph", type=int, default=156)
    border.add_argument("--min-m", type=int, default=32)
    border.add_argument("--max-m", type=int, default=44)

    e2e_summary = subparsers.add_parser(
        "e2e-summary",
        help="Plot end-to-end means and confidence intervals",
    )
    _add_common_output_args(
        e2e_summary,
        title="GPT-OSS-120B Triton end-to-end screen",
    )
    e2e_summary.add_argument("--world-sizes", type=int, nargs="+", default=[2, 4, 8])
    e2e_summary.add_argument("--contrast", default=DEFAULT_CONTRAST)

    e2e_blocks = subparsers.add_parser(
        "e2e-blocks",
        help="Plot end-to-end paired changes by restart block",
    )
    _add_common_output_args(
        e2e_blocks,
        title="GPT-OSS-120B Triton block consistency",
    )
    e2e_blocks.add_argument("--world-size", type=int, default=4)
    e2e_blocks.add_argument("--contrast", default=DEFAULT_CONTRAST)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        summary = load_json(args.input)
        if args.command == "graph-overview":
            selected = select_graph_comparisons(
                summary,
                world_sizes=args.world_sizes,
                calls_per_graph=args.calls_per_graph,
                min_m=args.min_m,
                max_m=args.max_m,
            )
            plot_graph_overview(
                selected,
                output=args.output,
                title=args.title,
                caption=args.caption,
            )
        elif args.command == "graph-border":
            selected = select_graph_comparisons(
                summary,
                world_sizes=[args.world_size],
                calls_per_graph=args.calls_per_graph,
                min_m=args.min_m,
                max_m=args.max_m,
            )
            plot_graph_border(
                selected[args.world_size],
                output=args.output,
                title=args.title,
                caption=args.caption,
            )
        elif args.command == "e2e-summary":
            metrics = extract_e2e_metrics(
                summary,
                world_sizes=args.world_sizes,
                contrast=args.contrast,
            )
            plot_e2e_summary(
                metrics,
                output=args.output,
                title=args.title,
                caption=args.caption,
            )
        else:
            blocks = extract_e2e_blocks(
                summary,
                world_size=args.world_size,
                contrast=args.contrast,
            )
            plot_e2e_blocks(
                blocks,
                world_size=args.world_size,
                output=args.output,
                title=args.title,
                caption=args.caption,
            )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
