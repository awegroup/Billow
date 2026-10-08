# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# SPDX-License-Identifier: Apache-2.0

"""Figures for the hanging-kite validation: solved shape against the 3D markers.

The shape figures compare the solved leading edge, struts and canopy against
Haanen's photogrammetry markers. Each model is placed by its own rigid best fit
to the markers (``run_hanging_shape_comparison.fit_shape``), so none is drawn in
a frame chosen for another. The length figures reproduce the published scalar
comparison -- twelve lengths per case, only six of them independent and none a
height or a chordwise measure -- so that the numbers in the literature can be
checked, not because they can decide a shape.

``shape_error.png``      point-to-surface RMS per load case and per marker
                         group, for every model and the as-built null model.
``shape_overview.png``   front view of all ten cases, models over markers.
``shape_<case>.png``     front / top / side views of one case, plus the
                         distance from each marker to each model along the span.
``summary.png``          span and mean relative length error, every case.
``lengths.png``          the twelve measured lengths per case, model over
                         measured.
``canopy_<case>.png``    the solved kite in 3D with the markers overlaid.
``regime_<case>.png``    the same, canopy coloured by wrinkling regime.

Run from the project root, after the validation run::

    python validation/plot_hanging_validation.py
    python validation/plot_hanging_validation.py \\
        --results results/hanging/validation.npz results/hanging/validation_scaled.npz \\
        --labels "Billow" "Billow, EI x0.36"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

from billow.elements import SLACK, TAUT, WRINKLED, membrane_regimes
from billow.plotting import PALETTE, set_plot_style
from hanging_kite import (LENGTH_NAMES, LOAD_CASES, build_model, canopy_grid,
                          canopy_triangles, extract_lengths, kite_fem_lengths,
                          load_reference, measured_lengths)
from run_hanging_shape_comparison import (CENTRE_LOAD_CASES, GROUPS, MARKERS,
                                          case_label, score_all,
                                          shape_models)

#: Colour per model, fixed so a model keeps its colour whatever else is drawn.
MODEL_COLOURS = [PALETTE["Sky Blue"], PALETTE["Bluish Green"], PALETTE["Reddish Purple"]]
KFEM = PALETTE["Vermillion"]
BUILT = "#9aa0a6"
MARKER = PALETTE["Black"]
FABRIC = "#b9c4cc"
TUBE = "#1a1a1a"
CENTRE_SHADE = "#f2c14e"

VIEWS = {"front": (1, 2, "y (m)", "z (m)"),
         "top": (1, 0, "y (m)", "x (m)"),
         "side": (0, 2, "x (m)", "z (m)")}


def colour_of(label: str, labels: list[str]) -> str:
    if label == "kite_fem":
        return KFEM
    if label == "as built":
        return BUILT
    return MODEL_COLOURS[labels.index(label) % len(MODEL_COLOURS)]


def structure_lines(nodes: np.ndarray, grid: np.ndarray):
    """Leading edge and strut polylines -- the inflatable structure only."""
    leading = list(range(1, 58, 2))
    struts = [grid[row].tolist() for row in range(len(grid))
              if int(grid[row, 0]) in (1, 7, 13, 19, 25, 33, 39, 45, 51, 57)]
    return [nodes[leading]] + [nodes[s] for s in struts]


def case_title(case: int) -> str:
    pressure, _, _ = LOAD_CASES[case - 1]
    return f"case {case}: {pressure} bar, {case_label(case)}"


def displayed(case, fits, models, frame):
    """Every model's nodes, placed by its own fit and shown in ``frame``'s model frame.

    ``frame`` is one model's fit; its model frame is used for display only. A
    model's nodes go to the measurement frame through that model's fit, then
    back through ``frame``, so every model sits where it best matches the
    markers and all share one set of axes.
    """
    return {label: frame.to_model(fits[case][label].to_measurement(positions[case]))
            for label, positions in models if case in positions}


def draw_view(axis, view, shapes, markers, grid, labels, *, with_labels=True):
    i, j, xlabel, ylabel = VIEWS[view]
    order = [l for l in shapes if l == "as built"] + [l for l in shapes if l != "as built"]
    for label in order:
        built = label == "as built"
        for k, line in enumerate(structure_lines(shapes[label], grid)):
            axis.plot(line[:, i], line[:, j], color=colour_of(label, labels),
                      linewidth=0.9 if built else 1.4,
                      linestyle="--" if built else "-", solid_capstyle="round",
                      label=label if (k == 0 and with_labels) else None,
                      zorder=2 if built else 3)
    tube = np.vstack([points for group, points in markers.items() if group != "CAN"])
    axis.scatter(tube[:, i], tube[:, j], s=10, c=MARKER, linewidths=0, zorder=5,
                 label="markers, tubes" if with_labels else None)
    if "CAN" in markers:
        axis.scatter(markers["CAN"][:, i], markers["CAN"][:, j], s=12,
                     facecolors="none", edgecolors=MARKER, linewidths=0.8, zorder=5,
                     label="markers, canopy" if with_labels else None)
    if view in ("front", "side"):
        axis.invert_yaxis()                 # z downwards: the kite hangs
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)


def figure_case(case, fits, markers, models, grid, labels, output):
    frame = fits[case][labels[0]]
    shapes = displayed(case, fits, models, frame)
    shown = {g: frame.position[g] for g in markers[case]}

    figure = plt.figure(figsize=(12.0, 6.4))
    grid_spec = figure.add_gridspec(2, 3, height_ratios=(1.0, 0.8))
    axes = [figure.add_subplot(grid_spec[0, k]) for k in range(3)]
    for axis, view in zip(axes, VIEWS):
        draw_view(axis, view, shapes, shown, grid, labels, with_labels=view == "front")
        axis.set_title(view, fontsize=9)
    axes[0].legend(loc="lower center", fontsize=7, ncol=2)

    # Where along the span each model is wrong: marker-to-surface distance
    # against arc length along the MEASURED leading edge. Spanwise y would fold
    # back on itself where the tips curl in. Strut markers sit where their strut
    # meets the leading edge, canopy markers at the nearest leading-edge marker.
    span = figure.add_subplot(grid_spec[1, :])
    leading = frame.position["LE"]
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(leading, axis=0), axis=1))])
    arc -= 0.5 * arc[-1]

    def along(points):
        nearest = np.linalg.norm(points[:, None, :] - leading[None, :, :], axis=2)
        return arc[np.argmin(nearest, axis=1)]

    strut_groups = [g for g in sorted(frame.position) if g.startswith("strut")]
    def root(points):
        nearest = np.linalg.norm(points[:, None, :] - leading[None, :, :], axis=2)
        return arc[np.unravel_index(np.argmin(nearest), nearest.shape)[1]]

    strut_arc = np.concatenate([np.full(len(frame.position[g]), root(frame.position[g]))
                                for g in strut_groups])
    for number, label in enumerate(labels):
        fit = fits[case][label]
        colour = colour_of(label, labels)
        nudge = (number - (len(labels) - 1) / 2) * 0.04       # keep stacks apart
        span.plot(arc, fit.distance["LE"] * 1000, "-o", color=colour,
                  markersize=3.5, linewidth=1.2,
                  label=f"{label}  (RMS {fit.rms() * 1000:.0f} mm)")
        span.scatter(strut_arc + nudge, fit.distance["struts"] * 1000, s=12,
                     marker="s", color=colour, linewidths=0, alpha=0.8)
        if len(fit.distance["canopy"]):
            span.scatter(along(frame.position["CAN"]) + nudge,
                         fit.distance["canopy"] * 1000, s=16, marker="^",
                         facecolors="none", edgecolors=colour, linewidths=0.8)
    span.set_xlabel("position along the measured leading edge, from mid-span (m)")
    span.set_ylabel("distance to model surface (mm)")
    span.set_ylim(bottom=0)
    handles, names = span.get_legend_handles_labels()
    handles += [plt.Line2D([0], [0], marker=m, linestyle="none", color="#555555",
                           markerfacecolor=f, markersize=5)
                for m, f in (("o", "#555555"), ("s", "#555555"), ("^", "none"))]
    names += ["leading edge", "struts", "canopy"]
    span.legend(handles, names, fontsize=7, ncol=2, loc="upper right")

    figure.suptitle(case_title(case), fontsize=10)
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def figure_overview(fits, markers, models, grid, labels, output):
    cases = sorted(fits)
    figure, axes = plt.subplots(2, 5, figsize=(14.0, 5.2))
    for axis, case in zip(axes.ravel(), cases):
        frame = fits[case][labels[0]]
        shapes = displayed(case, fits, models, frame)
        shown = {g: frame.position[g] for g in markers[case]}
        draw_view(axis, "front", shapes, shown, grid, labels, with_labels=case == cases[0])
        axis.set_title(case_title(case)
                       + "\nRMS " + ", ".join(f"{fits[case][l].rms() * 1000:.0f}"
                                              for l in labels) + " mm", fontsize=8)
        if case in CENTRE_LOAD_CASES:
            axis.set_facecolor(matplotlib.colors.to_rgba(CENTRE_SHADE, 0.10))
        axis.tick_params(labelsize=7)
        axis.set_xlabel(axis.get_xlabel(), fontsize=8)
        axis.set_ylabel(axis.get_ylabel(), fontsize=8)
    for axis in axes.ravel()[len(cases):]:
        axis.set_visible(False)
    handles, names = axes.ravel()[0].get_legend_handles_labels()
    figure.legend(handles, names, loc="lower center", ncol=len(names), fontsize=8)
    figure.suptitle("Front view, every model by its own rigid fit to the markers; "
                    "RMS in legend order (" + ", ".join(labels) + ")", fontsize=10)
    figure.tight_layout(rect=(0, 0.05, 1, 0.96))
    figure.savefig(output)
    plt.close(figure)


def figure_error(fits, labels, output):
    cases = sorted(fits)
    index = np.arange(len(cases))
    width = 0.8 / len(labels)
    offsets = (np.arange(len(labels)) - (len(labels) - 1) / 2) * width

    figure, (left, right) = plt.subplots(
        1, 2, figsize=(12.0, 3.8), gridspec_kw={"width_ratios": (2.4, 1.0)})
    for position, case in zip(index, cases):
        if case in CENTRE_LOAD_CASES:
            left.axvspan(position - 0.5, position + 0.5, color=CENTRE_SHADE,
                         alpha=0.18, zorder=0, linewidth=0)
    for offset, label in zip(offsets, labels):
        left.bar(index + offset, [fits[c][label].rms() * 1000 for c in cases],
                 width * 0.92, color=colour_of(label, labels), label=label)
    left.set_xticks(index)
    left.set_xticklabels([f"{c}\n" + case_label(c).replace(" ", "\n", 1)
                          for c in cases], fontsize=7)
    left.set_xlabel("load case  (shaded: centre load)")
    left.set_ylabel("RMS marker-to-surface\ndistance, tubes (mm)")
    left.legend(fontsize=7, ncol=len(labels), loc="upper right")
    left.grid(axis="x", visible=False)

    # Which part of the kite carries the error, over the ten cases.
    group_index = np.arange(len(GROUPS))
    for offset, label in zip(offsets, labels):
        values = []
        for group in GROUPS:
            pooled = np.concatenate([fits[c][label].distance[group] for c in cases])
            values.append(np.sqrt(np.mean(pooled**2)) * 1000 if len(pooled) else np.nan)
        right.bar(group_index + offset, values, width * 0.92,
                  color=colour_of(label, labels))
    right.set_xticks(group_index)
    right.set_xticklabels(["leading edge", "struts", "canopy"])
    right.set_ylabel("RMS over all ten cases (mm)")
    right.grid(axis="x", visible=False)

    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def length_series(models, cases):
    """The twelve lengths per case for every solved model and kite_fem."""
    published = kite_fem_lengths()
    series = []
    for label, positions in models:
        if label == "as built":
            continue
        if label == "kite_fem":       # its published table, not re-extracted
            series.append((label, published[np.asarray(cases) - 1]))
        elif all(c in positions for c in cases):
            series.append((label, np.array([extract_lengths(positions[c]) for c in cases])))
    return series


def figure_lengths(cases, series, measured, labels, output):
    """The twelve measured quantities per case, model over measured."""
    figure, axes = plt.subplots(2, 5, figsize=(14.0, 5.4), sharex=True)
    index = np.arange(len(LENGTH_NAMES))
    for column, (axis, case) in enumerate(zip(axes.ravel(), cases)):
        axis.plot(index, measured[column], "o-", color=MARKER, markersize=3,
                  linewidth=1.2, label="measured")
        for label, rows in series:
            axis.plot(index, rows[column], "^--", color=colour_of(label, labels),
                      markersize=3, linewidth=1.0, label=label)
        axis.set_yscale("log")
        axis.set_title(case_title(case), fontsize=8)
        axis.set_xticks(index)
        axis.set_xticklabels(LENGTH_NAMES, rotation=90, fontsize=6)
        if column == 0:
            axis.set_ylabel("length (m)")
            axis.legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def figure_summary(cases, series, measured, labels, output, built_span):
    """Span and mean relative length error across the cases."""
    figure, (left, right) = plt.subplots(1, 2, figsize=(11.0, 3.8))
    index = np.arange(len(cases))
    for axis in (left, right):
        for position, case in zip(index, cases):
            if case in CENTRE_LOAD_CASES:
                axis.axvspan(position - 0.5, position + 0.5, color=CENTRE_SHADE,
                             alpha=0.18, zorder=0, linewidth=0)

    bars = [("measured", measured, MARKER)] + [
        (label, rows, colour_of(label, labels)) for label, rows in series]
    width = 0.8 / len(bars)
    offsets = (np.arange(len(bars)) - (len(bars) - 1) / 2) * width
    for offset, (label, rows, colour) in zip(offsets, bars):
        left.bar(index + offset, rows[:, 9], width * 0.92, color=colour, label=label)
    left.axhline(built_span, color=BUILT, linestyle="--", linewidth=1.0, label="as built")
    left.set_ylabel("span (m)")
    left.set_ylim(0, 1.3 * built_span)          # headroom for the legend

    width = 0.8 / len(series)
    offsets = (np.arange(len(series)) - (len(series) - 1) / 2) * width
    for offset, (label, rows) in zip(offsets, series):
        error = np.abs((rows - measured) / measured).mean(axis=1) * 100
        right.bar(index + offset, error, width * 0.92, color=colour_of(label, labels),
                  label=label)
    right.set_ylabel("mean relative length error (%)")

    for axis in (left, right):
        axis.set_xticks(index)
        axis.set_xticklabels(cases)
        axis.set_xlabel("load case  (shaded: centre load)")
        axis.grid(axis="x", visible=False)
    left.legend(fontsize=7, ncol=3, loc="upper left")
    right.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


REGIME_COLOUR = {TAUT: "#2b7bba", WRINKLED: "#f2c14e", SLACK: "#c0392b"}
REGIME_NAME = {TAUT: "taut", WRINKLED: "wrinkled", SLACK: "slack"}


def shade(faces, base, light=(0.35, -0.55, -0.75), ambient=0.45):
    """Lambertian shading per triangle, so the surface reads as a solid object."""
    normals = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.where(lengths < 1e-15, 1.0, lengths)
    direction = np.asarray(light, dtype=float)
    direction /= np.linalg.norm(direction)
    intensity = ambient + (1.0 - ambient) * np.abs(normals @ direction)
    rgb = np.asarray(matplotlib.colors.to_rgb(base))
    return np.clip(intensity[:, None] * rgb[None, :], 0.0, 1.0)


def figure_canopy(case, positions, grid, reference, output, markers=None,
                  pressure=0.15, elev=14.0, azim=-22.0, style="fabric"):
    """The deformed kite in 3D.

    ``style="fabric"`` renders a shaded canopy with the markers, placed by this
    model's own fit. ``style="regime"`` colours each triangle by its
    tension-field state instead -- taut, wrinkled or slack -- which is a
    physical prediction Billow makes and a spring net cannot.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    faces = positions[canopy_triangles(grid)]
    regimes = None
    if style == "regime":
        model, _, _ = build_model(reference, grid, pressure=pressure,
                                  canopy_stiffness=5000.0)
        regimes = membrane_regimes(positions, model.element_set("canopy"))["regime"]
        colours = [REGIME_COLOUR.get(int(r), "#999999") for r in regimes]
        edge, width = "#ffffff", 0.15
    else:
        colours = shade(faces, FABRIC)
        edge, width = "none", 0.0

    figure = plt.figure(figsize=(10.0, 5.4))
    axis = figure.add_subplot(111, projection="3d")
    axis.add_collection3d(Poly3DCollection(
        faces, facecolors=colours, edgecolors=edge, linewidths=width,
        alpha=1.0 if style == "fabric" else 0.92))
    for line in structure_lines(positions, grid):
        axis.plot(line[:, 0], line[:, 1], line[:, 2], color=TUBE,
                  linewidth=3.4, solid_capstyle="round", zorder=6)
    if markers:
        points = np.vstack(list(markers.values()))
        axis.scatter(points[:, 0], points[:, 1], points[:, 2], s=11, c=MARKER,
                     depthshade=False, zorder=10)

    handles = []
    if regimes is not None:
        counts = {k: int((regimes == k).sum()) for k in (TAUT, WRINKLED, SLACK)}
        handles = [plt.Line2D([0], [0], marker="s", linestyle="none", markersize=8,
                              markerfacecolor=REGIME_COLOUR[k], markeredgecolor="none",
                              label=f"{REGIME_NAME[k]} ({counts[k]})")
                   for k in (TAUT, WRINKLED, SLACK) if counts[k]]
    if markers:
        handles.append(plt.Line2D([0], [0], marker="o", linestyle="none",
                                  markersize=5, color=MARKER, label="measured markers"))
    if handles:
        axis.legend(handles=handles, loc="upper left", fontsize=8)

    everything = np.vstack([positions[grid.ravel()],
                            *([np.vstack(list(markers.values()))] if markers else [])])
    low, high = everything.min(axis=0), everything.max(axis=0)
    pad = 0.06 * (high - low).max()
    low, high = low - pad, high + pad
    axis.set_xlim(low[0], high[0])
    axis.set_ylim(low[1], high[1])
    axis.set_zlim(high[2], low[2])          # z downwards: the kite hangs
    axis.set_box_aspect(tuple(high - low))
    axis.set_xlabel("x (m)", labelpad=2)
    axis.set_ylabel("y (m)")
    axis.set_zlabel("z (m)")
    axis.xaxis.set_major_locator(plt.MaxNLocator(3))
    axis.tick_params(axis="x", labelsize=7, pad=-2)
    axis.view_init(elev=elev, azim=azim)
    axis.set_proj_type("ortho")
    axis.set_title(case_title(case)
                   + ("   |   canopy shaded by wrinkling regime" if style == "regime"
                      else "   |   solved shape, measured markers overlaid"), fontsize=10)
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, nargs="+",
                        default=[Path("results/hanging/validation.npz"),
                                 Path("results/hanging/validation_scaled.npz")],
                        help="solved runs; any not yet produced is skipped")
    parser.add_argument("--labels", nargs="+", default=["Billow", "Billow, EI x0.36"])
    parser.add_argument("--markers", type=Path, default=MARKERS)
    parser.add_argument("--canopy-cases", type=int, nargs="*", default=[1, 5],
                        help="cases to render in 3D with the canopy shaded")
    parser.add_argument("--canopy-from", type=int, default=-1,
                        help="index into the solved runs of the one rendered in 3D")
    parser.add_argument("--no-kite-fem", action="store_true")
    parser.add_argument("--figures", type=Path, default=Path("results/hanging/figures"))
    parser.add_argument("--no-title", action="store_true",
                        help="drop figure titles (for a paper, where the caption carries them)")
    arguments = parser.parse_args()
    if len(arguments.labels) != len(arguments.results):
        parser.error("give one --labels entry per --results file")
    set_plot_style()
    if arguments.no_title:
        import matplotlib.axes
        import matplotlib.figure
        matplotlib.figure.Figure.suptitle = lambda self, *a, **k: None
        matplotlib.axes.Axes.set_title = lambda self, *a, **k: None

    models = shape_models(arguments.results, arguments.labels,
                          kite_fem=not arguments.no_kite_fem)
    labels = [label for label, _ in models]
    fits, markers = score_all(models, arguments.markers)
    reference = load_reference()
    grid = canopy_grid(reference.nodes)
    arguments.figures.mkdir(parents=True, exist_ok=True)

    figure_error(fits, labels, arguments.figures / "shape_error.png")
    figure_overview(fits, markers, models, grid, labels,
                    arguments.figures / "shape_overview.png")
    for case in sorted(fits):
        figure_case(case, fits, markers, models, grid, labels,
                    arguments.figures / f"shape_{case}.png")

    cases = sorted(fits)
    series = length_series(models, cases)
    measured = measured_lengths()[np.asarray(cases) - 1]
    figure_summary(cases, series, measured, labels, arguments.figures / "summary.png",
                   built_span=extract_lengths(reference.nodes)[9])
    figure_lengths(cases, series, measured, labels, arguments.figures / "lengths.png")

    solved = [m for m in models if m[0] not in ("kite_fem", "as built")]
    rendered_label, rendered = solved[arguments.canopy_from % len(solved)]
    for case in arguments.canopy_cases:
        if case not in rendered or case not in fits:
            continue
        pressure = LOAD_CASES[case - 1][0]
        fit = fits[case][rendered_label]
        figure_canopy(case, rendered[case], grid, reference,
                      arguments.figures / f"canopy_{case}.png",
                      markers=fit.position, pressure=pressure, style="fabric")
        figure_canopy(case, rendered[case], grid, reference,
                      arguments.figures / f"regime_{case}.png",
                      markers=None, pressure=pressure, style="regime")
    print(f"  -> {arguments.figures}")


if __name__ == "__main__":
    main()
