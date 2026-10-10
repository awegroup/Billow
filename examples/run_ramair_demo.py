# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# SPDX-License-Identifier: Apache-2.0

"""A multi-cell ram-air canopy: fabric cells inflated by a uniform pressure.

A box canopy of ``n_cells`` cells side by side -- top and bottom skins,
leading- and trailing-edge closure walls, a rib on every cell boundary -- held
by bridle lines from the bottom skin to one tow point. The internal pressure
is ONE uniform gauge pressure over the outer envelope, applied as the
potential ``-pV`` of :class:`billow.elements.PressureKernel`; the internal ribs
see the same pressure on both faces and carry no net pressure load, so they
are plain membranes and appear in no pressure set at all. The skins are cut
2% oversize in span so they pillow between the ribs, as a real ram-air skin
does. The pillowing skins also pull the outer ribs inward harder than the
pressure on them pushes out, and nothing else restrains the span, so the box
contracts spanwise: a real canopy's span is set by the same balance.

Pressure alone has zero resultant: every bridle would hang slack and the
knots would be a mechanism (an unbalanced free body is reported, not hidden).
A dead lift on the top skin stands in for the aerodynamic load, so the
bridles go taut and carry it to the tow point. The pressure is ramped on one
solver build, each step warm-started from the last, through
``ElementSet.with_param_column("pressure", ...)``; the lift is ramped with it,
since both scale with the dynamic pressure, and a full lift on a barely
inflated canopy would crush the cells.

The acceptance tolerance is scaled by hand: the solver scales it with the
EXTERNAL forces only, and the pressure lives inside the energy.

Run from the project root::

    python examples/run_ramair_demo.py
    python examples/run_ramair_demo.py --show
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from billow.plotting import PALETTE
from billow import MinimumEnergySolver, StructuralModel
from billow.elements import (
    SLACK,
    TAUT,
    WRINKLED,
    build_cable_elements,
    build_membrane_elements,
    build_pressure_elements,
    enclosed_volume,
    membrane_regimes,
)

OUTPUT_DIR = Path("results/demo")

FABRIC_MODULUS = 5.0e8  # Pa
FABRIC_POISSON = 0.3
FABRIC_THICKNESS = 3.0e-4  # m
BRIDLE_STIFFNESS = 2.0e5  # N/m

REGIME_COLORS = ListedColormap(
    [PALETTE["Sky Blue"], PALETTE["Orange"], PALETTE["Vermillion"]]
)
REGIME_LABELS = {SLACK: "slack", WRINKLED: "wrinkled", TAUT: "taut"}


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def facing_away(nodes, triangles, centre):
    """Re-wind each triangle so its normal points away from ``centre``."""
    triangles = np.array(triangles, dtype=int)
    corners = nodes[triangles]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    inward = np.einsum("ij,ij->i", normals, corners.mean(axis=1) - centre) < 0.0
    triangles[inward] = triangles[inward][:, [0, 2, 1]]
    return triangles


def build_box_canopy(
    n_cells=6,
    chord=1.0,
    span=3.0,
    height=0.18,
    n_chord=11,
    n_span_per_cell=4,
    n_height=4,
    skin_slack=0.02,
):
    """Nodes, per-surface triangle groups, the outward envelope and per-cell lists.

    Nodes sit on a ``chord x span x height`` grid, but only where a surface
    passes: the skins and closure walls at every station, the ribs only on
    cell boundaries. Returns ``(nodes, groups, envelope, cells, rib_stations)``
    with ``groups`` holding ``top``, ``bottom``, ``ribs`` (a list), ``le`` and
    ``te``. The skins' reference geometry is ``nodes`` stretched by
    ``skin_slack`` in span, so they are cut oversize and pillow between ribs.
    """
    x_axis = np.linspace(0.0, chord, n_chord)
    y_axis = np.linspace(0.0, span, n_cells * n_span_per_cell + 1)
    z_axis = np.linspace(0.0, height, n_height)
    rib_stations = np.arange(0, len(y_axis), n_span_per_cell)

    nodes: list[list[float]] = []
    index: dict[tuple[int, int, int], int] = {}

    def node(i, j, k):
        key = (i, j, k)
        if key not in index:
            index[key] = len(nodes)
            nodes.append([x_axis[i], y_axis[j], z_axis[k]])
        return index[key]

    def quad_grid(corner, u_range, v_range):
        triangles = []
        for u in u_range:
            for v in v_range:
                a, b, c, d = corner(u, v), corner(u + 1, v), corner(u + 1, v + 1), corner(u, v + 1)
                triangles += [[a, b, c], [a, c, d]]
        return np.array(triangles)

    chordwise, spanwise, vertical = range(n_chord - 1), range(len(y_axis) - 1), range(n_height - 1)
    top = quad_grid(lambda i, j: node(i, j, n_height - 1), chordwise, spanwise)
    bottom = quad_grid(lambda i, j: node(i, j, 0), chordwise, spanwise)
    ribs = [quad_grid(lambda i, k, j=j: node(i, j, k), chordwise, vertical) for j in rib_stations]
    leading = quad_grid(lambda j, k: node(0, j, k), spanwise, vertical)
    trailing = quad_grid(lambda j, k: node(n_chord - 1, j, k), spanwise, vertical)
    nodes = np.array(nodes)

    box_centre = np.array([0.5 * chord, 0.5 * span, 0.5 * height])
    envelope = facing_away(
        nodes, np.vstack([top, bottom, ribs[0], ribs[-1], leading, trailing]), box_centre
    )

    def within(triangles, cell):
        y_mid = nodes[triangles][:, :, 1].mean(axis=1)
        lower, upper = y_axis[rib_stations[cell]], y_axis[rib_stations[cell + 1]]
        return triangles[(y_mid > lower) & (y_mid < upper)]

    cells = []
    for cell in range(n_cells):
        centre = box_centre.copy()
        centre[1] = 0.5 * (y_axis[rib_stations[cell]] + y_axis[rib_stations[cell + 1]])
        walls = [within(group, cell) for group in (top, bottom, leading, trailing)]
        cells.append(facing_away(nodes, np.vstack(walls + [ribs[cell], ribs[cell + 1]]), centre))

    groups = {"top": top, "bottom": bottom, "ribs": ribs, "le": leading, "te": trailing}
    reference = {"skins": nodes * np.array([1.0, 1.0 + skin_slack, 1.0])}
    return nodes, groups, envelope, cells, rib_stations, reference, index


def build_model(pressure, n_cells=6, chord=1.0, span=3.0, **kwargs):
    nodes, groups, envelope, cells, rib_stations, reference, index = build_box_canopy(
        n_cells=n_cells, chord=chord, span=span, **kwargs
    )
    skins_triangles = np.vstack([groups["top"], groups["bottom"]])
    walls_triangles = np.vstack(groups["ribs"] + [groups["le"], groups["te"]])
    skins = build_membrane_elements(
        reference["skins"], skins_triangles, FABRIC_THICKNESS, FABRIC_MODULUS,
        FABRIC_POISSON, name="skins",
    )
    walls = build_membrane_elements(
        nodes, walls_triangles, FABRIC_THICKNESS, FABRIC_MODULUS, FABRIC_POISSON, name="walls"
    )
    cell_pressure = build_pressure_elements(nodes, envelope, pressure, name="cell_pressure")

    # Bridles: from the bottom-skin rib lines at two chord stations to one tow point.
    n_chord = max(i for i, _, _ in index) + 1
    attachments = [
        index[(i, j, 0)] for j in rib_stations for i in (n_chord // 4, (3 * n_chord) // 4)
    ]
    tow = len(nodes)
    nodes = np.vstack([nodes, [0.5 * chord, 0.5 * span, -1.5 * span]])
    connectivity = np.array([[node, tow] for node in attachments])
    lengths = np.linalg.norm(nodes[attachments] - nodes[tow], axis=1)
    bridles = build_cable_elements(
        connectivity, lengths, np.full(len(lengths), BRIDLE_STIFFNESS), name="bridles"
    )

    model = StructuralModel(
        nodes, [skins, walls, cell_pressure, bridles], fixed_translation_nodes=[tow]
    )
    surfaces = {
        "top": ("skins", slice(0, len(groups["top"]))),
        "bottom": ("skins", slice(len(groups["top"]), len(skins_triangles))),
        "ribs": ("walls", slice(0, sum(len(rib) for rib in groups["ribs"]))),
        "closure walls": ("walls", slice(sum(len(rib) for rib in groups["ribs"]), len(walls_triangles))),
    }
    return model, groups, cells, surfaces, index, rib_stations, tow


def pillow_height(positions, index, rib_stations, n_span_per_cell=4):
    """Mean rise of the top skin at mid-cell above the rib tops, at mid-chord."""
    n_chord = max(i for i, _, _ in index) + 1
    k_top = max(k for _, _, k in index)
    i = n_chord // 2
    rises = []
    for left, right in zip(rib_stations[:-1], rib_stations[1:]):
        mid = positions[index[(i, (left + right) // 2, k_top)], 2]
        ribs = 0.5 * (positions[index[(i, left, k_top)], 2] + positions[index[(i, right, k_top)], 2])
        rises.append(mid - ribs)
    return float(np.mean(rises))


def span_contraction(positions, reference, index, rib_stations):
    """Distance between the outer ribs at mid-chord, as a fraction of the reference."""
    n_chord = max(i for i, _, _ in index) + 1
    k_top = max(k for _, _, k in index)
    outer = [index[(n_chord // 2, j, k_top)] for j in (rib_stations[0], rib_stations[-1])]
    return float(
        np.linalg.norm(positions[outer[1]] - positions[outer[0]])
        / np.linalg.norm(reference[outer[1]] - reference[outer[0]])
    )


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def draw_surface(axes, positions, triangles, facevalues, cmap, norm, alpha=0.95):
    polygons = Poly3DCollection(
        [positions[triangle] for triangle in triangles],
        alpha=alpha, edgecolor="0.35", linewidths=0.15,
    )
    polygons.set_facecolor(cmap(norm(facevalues)))
    axes.add_collection3d(polygons)
    return polygons


def equalize(axes, positions, pad=0.05):
    lower, upper = positions.min(axis=0), positions.max(axis=0)
    centre = 0.5 * (lower + upper)
    radius = 0.5 * (upper - lower).max() * (1.0 + pad)
    axes.set_xlim(centre[0] - radius, centre[0] + radius)
    axes.set_ylim(centre[1] - radius, centre[1] + radius)
    axes.set_zlim(centre[2] - radius, centre[2] + radius)
    axes.set_box_aspect((1.0, 1.0, 1.0))
    axes.set_xlabel("x (m)")
    axes.set_ylabel("y (m)")
    axes.set_zlabel("z (m)")


# ---------------------------------------------------------------------------
# The case
# ---------------------------------------------------------------------------


def run(pressures=(60.0, 120.0, 180.0, 240.0, 300.0), lift=300.0, n_cells=6):
    """Ramp the pressure, and the lift with it, on one solver build.

    Both loads scale with the dynamic pressure of the flow, so the lift is
    ramped in proportion: a full lift on a barely inflated canopy would crush
    the cells before the pressure has stiffened them.
    """
    model, groups, cells, surfaces, index, rib_stations, tow = build_model(pressures[0], n_cells)
    pressure_set = model.element_set("cell_pressure")
    n_envelope = pressure_set.n_elements
    nodes = model.nodes

    # The dead lift that stands in for the aerodynamic load, on the top skin,
    # at the largest pressure; scaled down with the pressure during the ramp.
    top_nodes = np.unique(groups["top"])
    full_lift = np.zeros_like(nodes)
    full_lift[top_nodes, 2] = lift / len(top_nodes)

    # Pressure is inside the energy, so the solver's load scaling never sees
    # it: accept at 1e-6 of a nodal pressure force at the largest pressure.
    envelope_area = 0.5 * np.linalg.norm(np.cross(
        nodes[pressure_set.connectivity[:, 1]] - nodes[pressure_set.connectivity[:, 0]],
        nodes[pressure_set.connectivity[:, 2]] - nodes[pressure_set.connectivity[:, 0]],
    ), axis=1).sum()
    tolerance = 1e-6 * max(pressures) * envelope_area / model.n_nodes

    started = time.perf_counter()
    solver = MinimumEnergySolver(model, tolerance=1e-9, force_tolerance=tolerance)
    build_time = time.perf_counter() - started
    print(f"  {model.n_nodes} nodes, {model.layout.n_dof} DOF, {n_envelope} envelope triangles, "
          f"{len(cells)} cells, build {build_time:.2f} s, force tolerance {tolerance:.1e} N, "
          f"lift {lift:.0f} N at {max(pressures):.0f} Pa")

    reference_volume = np.array([enclosed_volume(nodes, cell) for cell in cells])
    history = []
    state = None
    for pressure in pressures:
        ramped = model.replaced(
            pressure_set.with_param_column("pressure", np.full(n_envelope, pressure))
        )
        started = time.perf_counter()
        forces = full_lift * (pressure / max(pressures))
        solution = solver.solve(forces, state=state, model=ramped)
        elapsed = time.perf_counter() - started
        state = solution.state
        positions = state.positions

        volumes = np.array([enclosed_volume(positions, cell) for cell in cells])
        regimes = {
            name: membrane_regimes(positions, ramped.element_set(set_name))["regime"][rows]
            for name, (set_name, rows) in surfaces.items()
        }
        counts = {
            name: {label: int((regime == code).sum()) for code, label in REGIME_LABELS.items()}
            for name, regime in regimes.items()
        }
        pillow = pillow_height(positions, index, rib_stations)
        span = span_contraction(positions, nodes, index, rib_stations)
        history.append({
            "pressure": pressure, "volumes": volumes, "pillow": pillow,
            "iterations": solution.iterations, "residual": solution.residual_norm,
            "converged": solution.converged, "regimes": regimes,
        })
        print(f"  p = {pressure:5.0f} Pa: {solution.iterations:3d} it, {elapsed:.2f} s, "
              f"residual {solution.residual_norm:.1e} N, converged={solution.converged} "
              f"({solution.status})")
        print(f"      cell volumes / reference: "
              f"{np.array2string(volumes / reference_volume, precision=3)}")
        print(f"      pillow height at mid-cell {pillow * 1e3:.1f} mm, span {span:.3f} of reference")
        for name, count in counts.items():
            print(f"      {name:14s} taut {count['taut']:4d}  wrinkled {count['wrinkled']:4d}  "
                  f"slack {count['slack']:4d}")
    return model, groups, cells, history, state, tow


def case_ramair():
    model, groups, cells, history, state, tow = run()
    positions = state.positions
    final = history[-1]

    figure = plt.figure(figsize=(12.5, 5.4))
    figure.suptitle(
        "Ram-air canopy: six cells under one uniform pressure, pressure ramped on one "
        "solver build",
        fontsize=12,
    )
    left = figure.add_subplot(1, 2, 1, projection="3d")
    norm = Normalize(-0.5, 2.5)
    skins = np.vstack([groups["top"], groups["bottom"]])
    walls = np.vstack(groups["ribs"] + [groups["le"], groups["te"]])
    skin_regimes = np.concatenate([final["regimes"]["top"], final["regimes"]["bottom"]])
    wall_regimes = np.concatenate([final["regimes"]["ribs"], final["regimes"]["closure walls"]])
    draw_surface(left, positions, skins, skin_regimes, REGIME_COLORS, norm, alpha=0.9)
    draw_surface(left, positions, walls, wall_regimes, REGIME_COLORS, norm, alpha=0.6)
    # The tow point sits 4.5 m below: frame the canopy and let the bridles
    # run out of the box towards it.
    canopy = positions[:tow]
    for node in model.element_set("bridles").connectivity[:, 0]:
        segment = np.vstack([positions[node], positions[node] + 0.25 * (positions[tow] - positions[node])])
        left.plot(*segment.T, color=PALETTE["Bluish Green"], linewidth=0.9)
    left.plot([], [], color=PALETTE["Bluish Green"], label="bridles (to the tow point)")
    equalize(left, canopy)
    left.view_init(elev=22, azim=-55)
    left.set_title(
        f"{model.layout.n_dof} DOF, p = {final['pressure']:.0f} Pa, "
        f"{final['iterations']} iterations, residual {final['residual']:.1e} N",
        fontsize=9,
    )
    handles = [
        plt.Line2D([], [], marker="s", linestyle="", markersize=9,
                   color=REGIME_COLORS(norm(code)), label=name)
        for code, name in REGIME_LABELS.items()
    ]
    left.legend(handles=handles + left.get_legend_handles_labels()[0],
                loc="upper left", frameon=False, fontsize=8)

    right = figure.add_subplot(1, 2, 2)
    pressures = [step["pressure"] for step in history]
    reference = np.array([enclosed_volume(model.nodes, cell) for cell in cells])
    mean_volume = [step["volumes"].sum() / reference.sum() for step in history]
    right.plot(pressures, mean_volume, "o-", color=PALETTE["Vermillion"],
               label="canopy volume / reference")
    right.set_xlabel("internal pressure (Pa)")
    right.set_ylabel("volume / reference")
    twin = right.twinx()
    twin.bar(pressures, [step["iterations"] for step in history], width=25.0,
             color=PALETTE["Sky Blue"], alpha=0.5, label="IPOPT iterations")
    twin.set_ylabel("IPOPT iterations per step")
    twin.grid(False)
    lines = right.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    right.legend(handles=lines, loc="upper left", frameon=False, fontsize=8)
    right.set_title(
        f"pillow height {final['pillow'] * 1e3:.0f} mm at mid-cell, "
        f"all steps converged={all(step['converged'] for step in history)}",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="open the figure")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no-title", action="store_true",
                        help="drop figure titles (for a paper, where the caption carries them)")
    arguments = parser.parse_args()
    if arguments.no_title:
        import matplotlib.figure, matplotlib.axes
        matplotlib.figure.Figure.suptitle = lambda self, *a, **k: None

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    print("[ramair]")
    started = time.perf_counter()
    figure = case_ramair()
    target = arguments.output_dir / "ramair.png"
    figure.savefig(target, dpi=160)
    print(f"  -> {target}  ({time.perf_counter() - started:.1f} s)")
    if arguments.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
