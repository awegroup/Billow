# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# SPDX-License-Identifier: Apache-2.0

"""Score solved kite shapes against the 3D photogrammetry markers.

The twelve published lengths are only six independent numbers, with no height
and no chordwise measure among them, so they cannot decide whether a model has
the right *shape*. Haanen's marker clouds can: 60--76 three-dimensional points
per load case on the leading edge, the eight struts and the canopy, vendored
under ``data/hanging_validation/markers/``.

Every model is scored the same way -- Billow, ``kite_fem``'s published states,
and the undeformed as-built geometry as the null model. Anything not beating the
null model has explained nothing.

Alignment
---------
The markers are in the stereo rig's frame, so model and measurement must be
brought together before anything can be compared. Only a *rigid* transform is
allowed --- rotation and translation, never scale --- because scale is exactly
the thing under test. Each model gets its OWN best fit, so no model is scored in
a frame chosen for another.

Correspondence comes from structure, not from fitting: measured ``strut0``
through ``strut7`` are ordered across the span and their first marker sits at
the leading edge, so their roots pair with the model's eight strut leading-edge
nodes. That gives eight anchors for an initial Kabsch fit, tried in both
spanwise orderings because the marker numbering direction is not documented.
The fit is then refined by ICP against the model tube axes (each marker to the
nearest point on its own polyline, so no marker is assumed to sit on a node),
and finished by Gauss-Newton on the same marker-to-surface distance that is
scored; ``_align`` says why it takes both. The canopy markers take no part in the
fit; they are scored afterwards against the canopy surface, as an independent
check.

What is reported
----------------
Distance from each marker to the model surface: to the tube *surface* for the
leading-edge and strut markers (they are stuck to the outside of the tubes,
while the model polyline is the tube axis, 55--100 mm away), and to the
triangulated canopy for the canopy markers. It does not require a marker to
correspond to a node, and it cannot be reduced by sliding markers along the
structure.

Run from the project root, after ``run_hanging_validation.py``::

    python validation/run_hanging_shape_comparison.py
    python validation/run_hanging_shape_comparison.py \\
        --results results/hanging/validation.npz results/hanging/validation_scaled.npz \\
        --labels "Billow" "Billow, EI x0.36"
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hanging_kite import (DATA, LOAD_CASES, canopy_grid, canopy_triangles,
                          kite_fem_state, load_reference)

MARKERS = DATA / "markers"

#: Measured file stem for each of the ten load cases, in case order.
CASE_FILES = ("P1_S", "P1_PL1", "P1_PL2", "P1_TL1", "P1_TL2",
              "P2_S", "P2_PL1", "P2_PL2", "P2_TL1", "P2_TL2")

#: The eight real struts (the model's two tip closures are not instrumented).
STRUT_LE_NODES = (7, 13, 19, 25, 33, 39, 45, 51)

#: Marker groups as scored: tube markers (which drive the fit) and canopy.
GROUPS = ("LE", "struts", "canopy")

#: Cases with a concentrated load on the mid-span leading edge.
CENTRE_LOAD_CASES = (2, 3, 7, 8)


def load_markers(stem: str, directory: Path = MARKERS) -> dict[str, np.ndarray]:
    """Marker positions by group, each ordered by ``idx_in_group``."""
    with open(directory / f"{stem}.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    groups: dict[str, list] = {}
    for row in rows:
        groups.setdefault(row["group"], []).append(
            (int(row["idx_in_group"]),
             float(row["x"]), float(row["y"]), float(row["z"]))
        )
    return {name: np.array([p[1:] for p in sorted(points)])
            for name, points in groups.items()}


def model_curves(positions: np.ndarray, grid: np.ndarray) -> dict[str, np.ndarray]:
    """Leading-edge and strut polylines from a solved configuration."""
    sections = {int(grid[row, 0]): grid[row] for row in range(len(grid))}
    curves = {"LE": positions[list(range(1, 58, 2))]}
    for index, node in enumerate(STRUT_LE_NODES):
        curves[f"strut{index}"] = positions[sections[node]]
    return curves


def curve_radii(reference, grid) -> dict[str, np.ndarray]:
    """Tube radius at each polyline node.

    Radii come from the beam elements; a node takes the mean of the elements
    meeting there, and nodes with no beam (canopy interior of a strut section)
    inherit the nearest beam radius along the section.
    """
    diameter = {}
    for (node_a, node_b), d in zip(reference.beams, reference.beam_diameter):
        diameter.setdefault(int(node_a), []).append(d)
        diameter.setdefault(int(node_b), []).append(d)
    radius_of = {n: 0.5 * float(np.mean(v)) for n, v in diameter.items()}

    sections = {int(grid[row, 0]): grid[row] for row in range(len(grid))}
    radii = {"LE": np.array([radius_of.get(n, 0.0) for n in range(1, 58, 2)])}
    for index, node in enumerate(STRUT_LE_NODES):
        values = np.array([radius_of.get(int(n), np.nan) for n in sections[node]])
        known = np.where(~np.isnan(values))[0]
        if len(known):
            for j in np.where(np.isnan(values))[0]:
                values[j] = values[known[np.argmin(np.abs(known - j))]]
        else:
            values[:] = 0.0
        radii[f"strut{index}"] = values
    return radii


def project_to_polyline(points: np.ndarray, polyline: np.ndarray,
                        radii: np.ndarray | None = None):
    """Closest point on a polyline, and optionally the tube radius there."""
    start, end = polyline[:-1], polyline[1:]
    segment = end - start
    length2 = np.maximum(np.einsum("ij,ij->i", segment, segment), 1e-15)

    delta = points[:, None, :] - start[None, :, :]
    t = np.clip(np.einsum("ijk,jk->ij", delta, segment) / length2[None, :], 0.0, 1.0)
    closest = start[None, :, :] + t[:, :, None] * segment[None, :, :]
    distance = np.linalg.norm(points[:, None, :] - closest, axis=2)
    pick = np.argmin(distance, axis=1)
    nearest = closest[np.arange(len(points)), pick]
    if radii is None:
        return nearest
    edge = 0.5 * (radii[:-1] + radii[1:])
    return nearest, edge[pick]


def distance_to_mesh(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Unsigned distance from each point to a triangle soup ``(n, 3, 3)``.

    The closest point on a triangle is either the in-plane projection, when that
    falls inside, or lies on one of its three edges.
    """
    a, b, c = faces[:, 0], faces[:, 1], faces[:, 2]
    normal = np.cross(b - a, c - a)
    area2 = np.maximum(np.einsum("ij,ij->i", normal, normal), 1e-30)
    result = np.empty(len(points))
    for k, p in enumerate(points):
        height = np.einsum("ij,ij->i", p - a, normal) / area2
        foot = p - height[:, None] * normal
        # barycentric sign tests on the foot of the perpendicular
        inside = ((np.einsum("ij,ij->i", np.cross(b - a, foot - a), normal) >= 0)
                  & (np.einsum("ij,ij->i", np.cross(c - b, foot - b), normal) >= 0)
                  & (np.einsum("ij,ij->i", np.cross(a - c, foot - c), normal) >= 0))
        best = np.where(inside, np.abs(height) * np.sqrt(area2), np.inf)
        for start, end in ((a, b), (b, c), (c, a)):
            edge = end - start
            t = np.clip(np.einsum("ij,ij->i", p - start, edge)
                        / np.maximum(np.einsum("ij,ij->i", edge, edge), 1e-30), 0.0, 1.0)
            best = np.minimum(best, np.linalg.norm(p - (start + t[:, None] * edge), axis=1))
        result[k] = best.min()
    return result


def kabsch(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rigid transform taking ``source`` onto ``target``; no scaling, no reflection."""
    centre_s, centre_t = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - centre_s).T @ (target - centre_t))
    correction = np.diag([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T))])
    rotation = vt.T @ correction @ u.T
    return rotation, centre_t - rotation @ centre_s


@dataclass(frozen=True)
class Fit:
    """One model's rigid best fit to one marker cloud.

    ``rotation`` and ``translation`` take MODEL coordinates to the measurement
    frame. ``distance`` holds the per-marker surface distance by group, and
    ``position`` the same markers carried into the model frame, for plotting.
    """

    rotation: np.ndarray
    translation: np.ndarray
    reverse: bool
    distance: dict[str, np.ndarray]
    position: dict[str, np.ndarray]

    def to_model(self, points: np.ndarray) -> np.ndarray:
        return (points - self.translation) @ self.rotation

    def to_measurement(self, points: np.ndarray) -> np.ndarray:
        return points @ self.rotation.T + self.translation

    def rms(self, groups=("LE", "struts")) -> float:
        values = np.concatenate([self.distance[g] for g in groups if len(self.distance[g])])
        return float(np.sqrt(np.mean(values**2)))


def _tube_pairs(markers, curves, radii, reverse):
    order = range(7, -1, -1) if reverse else range(8)
    pairs = [("LE", markers["LE"], curves["LE"], radii["LE"])] if "LE" in markers else []
    for measured_index, model_index in enumerate(order):
        name = f"strut{measured_index}"
        if name in markers and len(markers[name]):
            pairs.append((name, markers[name], curves[f"strut{model_index}"],
                          radii[f"strut{model_index}"]))
    return pairs


def _rodrigues(omega: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(omega)
    if angle < 1e-15:
        return np.eye(3)
    k = omega / angle
    cross = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross


def _align(markers, curves, radii, reverse, icp=20, newton=40):
    """Kabsch on the strut roots, ICP to the tube axes, then Gauss-Newton.

    Two stages because the two distances fail in opposite ways. Distance to
    the tube AXIS has a single basin, but minimising it biases every model
    toward the markers by up to a tube radius, since the markers all sit on the
    outside. Distance to the tube SURFACE is the quantity scored, but it has a
    second zero on the far side of every tube, a diameter away, which a rough
    start can lock onto. So ICP to the axes places the model, and Gauss-Newton
    on the signed marker-to-surface distance -- linearised about the current
    closest points -- removes the radius bias from there.
    """
    pairs = _tube_pairs(markers, curves, radii, reverse)
    roots = [(measured[0], curve[0]) for name, measured, curve, _ in pairs
             if name != "LE"]
    rotation, translation = kabsch(np.array([model for _, model in roots]),
                                   np.array([measured for measured, _ in roots]))
    targets = np.vstack([measured for _, measured, _, _ in pairs])
    for _ in range(icp):
        projected = np.vstack([
            project_to_polyline(measured, curve @ rotation.T + translation)
            for _, measured, curve, _ in pairs])
        rotation, translation = kabsch((projected - translation) @ rotation, targets)

    def linearise(rotation, translation):
        axis_points, gaps, normals = [], [], []
        for _, measured, curve, radius in pairs:
            axis, edge = project_to_polyline(measured, curve @ rotation.T + translation,
                                             radius)
            offset = measured - axis
            length = np.maximum(np.linalg.norm(offset, axis=1), 1e-12)
            axis_points.append(axis)
            gaps.append(length - edge)
            normals.append(offset / length[:, None])
        return np.vstack(axis_points), np.concatenate(gaps), np.vstack(normals)

    axis_points, gaps, normals = linearise(rotation, translation)
    for _ in range(newton):
        centre = axis_points.mean(axis=0)
        # moving the axis by delta + omega x (a - centre) shortens the gap by
        # n . delta + ((a - centre) x n) . omega
        jacobian = np.hstack([np.cross(axis_points - centre, normals), normals])
        step, *_ = np.linalg.lstsq(jacobian, gaps, rcond=None)
        # Backtrack: the markers are far from any model on the hard cases, a
        # large-residual regime where an undamped Gauss-Newton step overshoots.
        for scale in 0.5 ** np.arange(12):
            turn = _rodrigues(scale * step[:3])
            trial = (turn @ rotation,
                     turn @ (translation - centre) + centre + scale * step[3:])
            trial_state = linearise(*trial)
            if np.sum(trial_state[1] ** 2) < np.sum(gaps ** 2):
                break
        else:
            break
        rotation, translation = trial
        axis_points, gaps, normals = trial_state
        if scale * np.linalg.norm(step) < 1e-10:
            break
    return rotation, translation, pairs


def fit_shape(positions, grid, markers, radii) -> Fit:
    """Best of the two spanwise orderings, by tube-marker RMS."""
    curves = model_curves(positions, grid)
    faces = positions[canopy_triangles(grid)]
    best = None
    for reverse in (False, True):
        rotation, translation, pairs = _align(markers, curves, radii, reverse)
        distance = {"LE": np.empty(0), "struts": np.empty(0), "canopy": np.empty(0)}
        for name, measured, curve, radius in pairs:
            closest, edge = project_to_polyline(
                measured, curve @ rotation.T + translation, radius)
            surface = np.abs(np.linalg.norm(measured - closest, axis=1) - edge)
            group = "LE" if name == "LE" else "struts"
            distance[group] = np.concatenate([distance[group], surface])
        if "CAN" in markers:
            placed = faces @ rotation.T + translation
            distance["canopy"] = distance_to_mesh(markers["CAN"], placed)
        position = {g: (points - translation) @ rotation for g, points in markers.items()}
        candidate = Fit(rotation, translation, reverse, distance, position)
        if best is None or candidate.rms() < best.rms():
            best = candidate
    return best


def case_label(case: int) -> str:
    _, tip, point = LOAD_CASES[case - 1]
    if tip:
        return f"tip {tip:g} kg"
    if point:
        return f"centre {point:g} kg"
    return "gravity"


def shape_models(results: list[Path], labels: list[str], *, kite_fem=True,
                 as_built=True):
    """Every model to score, as ``(label, {case: positions})``."""
    reference = load_reference()
    models = []
    for path, label in zip(results, labels):
        if not Path(path).exists():
            print(f"  skipping {label}: {path} not found")
            continue
        data = np.load(path, allow_pickle=True)
        models.append((label, {int(c): p for c, p in zip(data["cases"], data["positions"])}))
    if kite_fem:
        models.append(("kite_fem", {c: kite_fem_state(c) for c in range(1, 11)}))
    if as_built:
        models.append(("as built", {c: reference.nodes for c in range(1, 11)}))
    return models


def score_all(models, directory: Path = MARKERS):
    """``fits[case][label] -> Fit`` and the markers, for every case on disk."""
    reference = load_reference()
    grid = canopy_grid(reference.nodes)
    radii = curve_radii(reference, grid)
    fits, markers = {}, {}
    for index, stem in enumerate(CASE_FILES):
        case = index + 1
        if not (directory / f"{stem}.csv").exists():
            continue
        markers[case] = load_markers(stem, directory)
        fits[case] = {label: fit_shape(positions[case], grid, markers[case], radii)
                      for label, positions in models if case in positions}
    return fits, markers


def shape_table(fits, labels) -> str:
    """The LaTeX table of tube-marker RMS per case and model, in mm."""
    cases = sorted(fits)
    lines = [
        r"\begin{tabular}{rl " + "r" * len(labels) + " r}",
        r"\toprule",
        r"LC & load & " + " & ".join(label.replace("_", r"\_") for label in labels)
        + r" & markers \\",
        r"\midrule",
    ]
    for case in cases:
        count = sum(len(fits[case][labels[0]].distance[g]) for g in ("LE", "struts"))
        mark = r"$^{\dagger}$" if case in CENTRE_LOAD_CASES else ""
        lines.append(f"{case}{mark} & {case_label(case)} & "
                     + " & ".join(f"{fits[case][l].rms() * 1000:.0f}" for l in labels)
                     + f" & {count} " + r"\\")
    lines.append(r"\midrule")
    buckets = (("gravity only", lambda c: case_label(c) == "gravity"),
               ("centre load", lambda c: c in CENTRE_LOAD_CASES),
               ("tip load", lambda c: case_label(c).startswith("tip")),
               ("all ten", lambda c: True))
    for name, keep in buckets:
        chosen = [c for c in cases if keep(c)]
        lines.append(r"\multicolumn{2}{r}{mean, " + name + "} & "
                     + " & ".join(f"{np.mean([fits[c][l].rms() for c in chosen]) * 1000:.0f}"
                                  for l in labels) + r" & \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--markers", type=Path, default=MARKERS,
                        help="directory holding P1_S.csv etc.")
    parser.add_argument("--results", type=Path, nargs="+",
                        default=[Path("results/hanging/validation.npz"),
                                 Path("results/hanging/validation_scaled.npz")],
                        help="solved runs; any not yet produced is skipped")
    parser.add_argument("--labels", nargs="+", default=["Billow", "Billow, EI x0.36"])
    parser.add_argument("--table", type=Path, default=Path("docs/table_shape.tex"))
    arguments = parser.parse_args()
    if len(arguments.labels) != len(arguments.results):
        parser.error("give one --labels entry per --results file")

    models = shape_models(arguments.results, arguments.labels)
    labels = [label for label, _ in models]
    fits, markers = score_all(models, arguments.markers)

    width = max(12, *(len(l) + 2 for l in labels))
    print("  point-to-surface RMS on the tube markers (mm); canopy markers in brackets")
    print(f"{'case':>5} {'load':>15} " + "".join(f"{l:>{width + 7}}" for l in labels))
    for case in sorted(fits):
        cells = []
        for label in labels:
            fit = fits[case][label]
            canopy = fit.distance["canopy"]
            extra = f"({np.sqrt(np.mean(canopy**2)) * 1000:4.0f})" if len(canopy) else " " * 6
            cells.append(f"{fit.rms() * 1000:>{width}.0f} {extra}")
        print(f"{case:>5} {case_label(case):>15} " + "".join(cells))
    print("\n  mean over the ten cases:")
    for label in labels:
        print(f"     {label:>{width}}: {np.mean([fits[c][label].rms() for c in fits]) * 1000:6.0f} mm")

    arguments.table.parent.mkdir(parents=True, exist_ok=True)
    arguments.table.write_text(shape_table(fits, labels) + "\n", encoding="utf-8")
    print(f"\n  -> {arguments.table}")


if __name__ == "__main__":
    main()
