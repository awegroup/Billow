# Copyright (c) 2023-2026 Oriol Cayon
#
# SPDX-License-Identifier: Apache-2.0

"""Inflated membranes against closed forms: the uniform-pressure potential.

Three cases, each a figure, each with a reference that owes nothing to the
discretisation:

* ``balloon``  -- a closed icosphere of SVK fabric under internal pressure. The
  continuum sphere inflates by the stretch ``lambda`` that makes
  ``Pi(lambda) = pi R0^2 t E (lambda^2 - 1)^2 / (1 - nu) - 4/3 pi p R0^3 lambda^3``
  stationary, ``lambda^2 - a lambda - 1 = 0`` with ``a = p R0 (1 - nu) / (E t)``.
  One solver build runs the whole pressure sweep (``with_param_column``), and
  the mesh error against that closed form falls as ``O(h^2)``.
* ``hencky``   -- the Foppl-von Karman clamped circular membrane under uniform
  pressure (Hencky, 1915), solved here as a boundary-value problem with
  ``scipy.integrate.solve_bvp`` at ``nu = 0.3``: ``w0 / a = 0.65344 (pa/Eh)^(1/3)``,
  centre ``N_r = 0.4311 Eh (pa/Eh)^(2/3)``, edge ``N_r = 0.3330 Eh (pa/Eh)^(2/3)``.
  The textbook ``0.662`` is the uncorrected series value. The CST disk converges
  onto the BVP from above; its radial stress resultant, taken from the
  membrane's own Green strains, follows the BVP profile.
* ``cylinder`` -- an open cylinder between two rigid, pinned end rings. Away
  from the ends the hoop stretch is the plane-strain closed form
  ``lambda_theta^2 - 1 = 2 p R (1 - nu^2) / (E t)``; near them a membrane
  boundary layer of order ``R`` brings it down to the ring. There is no clean
  closed form at finite length, so this is a figure, not an assertion.

Run from the repository root::

    python validation/run_pressure_benchmarks.py
    python validation/run_pressure_benchmarks.py --case hencky --show
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_bvp

from billow.plotting import PALETTE
from billow import MinimumEnergySolver, StructuralModel
from billow.elements import (
    build_membrane_elements,
    build_pressure_elements,
    membrane_regimes,
)

OUTPUT_DIR = Path("results/validation")

MODULUS = 5.0e8  # Pa
POISSON = 0.3
THICKNESS = 3.0e-4  # m

COLOURS = {"billow": PALETTE["Vermillion"], "reference": PALETTE["Black"]}


# ---------------------------------------------------------------------------
# Meshes, wound outward
# ---------------------------------------------------------------------------


def icosphere(level: int, radius: float = 1.0):
    phi = 0.5 * (1.0 + np.sqrt(5.0))
    nodes = np.array(
        [[-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
         [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
         [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]], float,
    )
    nodes *= radius / np.linalg.norm(nodes[0])
    triangles = np.array(
        [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
         [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
         [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
         [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]]
    )
    for k, (a, b, c) in enumerate(triangles):
        if np.dot(nodes[a], np.cross(nodes[b], nodes[c])) < 0.0:
            triangles[k] = [a, c, b]

    nodes = list(nodes)
    for _ in range(level):
        midpoints: dict[tuple[int, int], int] = {}

        def midpoint(i: int, j: int) -> int:
            key = (min(i, j), max(i, j))
            if key not in midpoints:
                point = 0.5 * (nodes[i] + nodes[j])
                nodes.append(point * radius / np.linalg.norm(point))
                midpoints[key] = len(nodes) - 1
            return midpoints[key]

        refined = []
        for a, b, c in triangles:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            refined += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        triangles = np.array(refined)
    return np.array(nodes), triangles


def disk_mesh(n_rings: int, radius: float = 1.0):
    """Flat disk, ``6 k`` nodes on ring ``k``, centre at node 0, normal ``+z``."""
    nodes = [[0.0, 0.0, 0.0]]
    ring_start = [0]
    for k in range(1, n_rings + 1):
        ring_start.append(len(nodes))
        for m in range(6 * k):
            angle = 2.0 * np.pi * m / (6 * k)
            r = radius * k / n_rings
            nodes.append([r * np.cos(angle), r * np.sin(angle), 0.0])
    triangles = []
    for k in range(1, n_rings + 1):
        inner, outer = ring_start[k - 1], ring_start[k]
        n_inner, n_outer = 6 * (k - 1), 6 * k
        for sector in range(6):
            for m in range(k):
                o0 = outer + (sector * k + m) % n_outer
                o1 = outer + (sector * k + m + 1) % n_outer
                i0 = inner + ((sector * (k - 1) + m) % n_inner if n_inner else 0)
                triangles.append([i0, o0, o1])
                if m < k - 1:
                    i1 = inner + (sector * (k - 1) + m + 1) % n_inner
                    triangles.append([i0, o1, i1])
    rim = np.arange(ring_start[n_rings], len(nodes))
    return np.array(nodes), np.array(triangles), rim


def cylinder_mesh(radius: float, length: float, n_circumference: int):
    """Open cylinder along ``z`` with square-ish triangles, normal outward."""
    spacing = 2.0 * np.pi * radius / n_circumference
    n_axial = max(2, int(round(length / spacing))) + 1
    angles = 2.0 * np.pi * np.arange(n_circumference) / n_circumference
    stations = np.linspace(0.0, length, n_axial)
    nodes = np.array(
        [[radius * np.cos(a), radius * np.sin(a), z] for z in stations for a in angles]
    )

    def index(i, j):
        return i * n_circumference + j % n_circumference

    triangles = []
    for i in range(n_axial - 1):
        for j in range(n_circumference):
            # (theta, z) grid: e_theta x e_z points outward.
            triangles.append([index(i, j), index(i, j + 1), index(i + 1, j + 1)])
            triangles.append([index(i, j), index(i + 1, j + 1), index(i + 1, j)])
    rings = np.concatenate([np.arange(n_circumference), index(n_axial - 1, 0) + np.arange(n_circumference)])
    return nodes, np.array(triangles), rings, stations


def surface_area(nodes, triangles) -> float:
    corners = nodes[triangles]
    return float(0.5 * np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1
    ).sum())


def mean_edge_length(nodes, triangles) -> float:
    edges = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    return float(np.linalg.norm(nodes[edges[:, 1]] - nodes[edges[:, 0]], axis=1).mean())


# ---------------------------------------------------------------------------
# Solves
# ---------------------------------------------------------------------------


def pressure_tolerance(pressure: float, area: float, n_nodes: int) -> float:
    """The solver scales acceptance with EXTERNAL forces only; pressure is inside
    the energy, so the tolerance is set to 1e-6 of a nodal pressure force."""
    return 1e-6 * pressure * area / n_nodes


def inflated_model(nodes, triangles, pressure, *, pinned=(), closed=True, wrinkling=True):
    fabric = build_membrane_elements(
        nodes, triangles, THICKNESS, MODULUS, POISSON, wrinkling=wrinkling
    )
    pressure_set = build_pressure_elements(nodes, triangles, pressure, closed=closed)
    model = StructuralModel(nodes, [fabric, pressure_set], fixed_translation_nodes=list(pinned))
    tolerance = pressure_tolerance(pressure, surface_area(nodes, triangles), len(nodes))
    return model, MinimumEnergySolver(model, tolerance=1e-10, force_tolerance=tolerance)


def svk_sphere_stretch(coefficient: float) -> float:
    return 0.5 * (coefficient + np.sqrt(coefficient**2 + 4.0))


def hencky_bvp(load: float, poisson: float = POISSON, inner: float = 1e-4):
    """FvK clamped membrane under uniform pressure, ``a = 1``, ``E h = 1``.

    Unknowns ``(u, w, N_r)`` on ``r in [inner, 1]``: radial equilibrium
    ``(r N_r)' = N_theta``, vertical equilibrium ``N_r w' = -p r / 2``, and the
    compatibility ``u' = eps_r - w'^2 / 2`` with ``eps_theta = u / r``. The
    centre condition is isotropy, ``N_r = N_theta``; the rim is clamped,
    ``u = w = 0``. Returns the solution object plus the three headline numbers.
    """
    stiffness = 1.0 / (1.0 - poisson**2)

    def rhs(r, y):
        u, _, n_r = y
        slope = -load * r / (2.0 * n_r)
        eps_t = u / r
        eps_r = n_r / stiffness - poisson * eps_t
        n_t = stiffness * (eps_t + poisson * eps_r)
        return np.vstack([eps_r - 0.5 * slope**2, slope, (n_t - n_r) / r])

    def boundary(ya, yb):
        # Isotropic at the centre: N_r = N_theta = N gives eps = N (1 - nu) / Eh.
        eps_centre = ya[2] * (1.0 - poisson)
        return np.array([ya[0] - eps_centre * inner, yb[0], yb[1]])

    r = np.linspace(inner, 1.0, 2000)
    guess = np.vstack([
        0.01 * r * (1.0 - r),
        0.66 * load ** (1.0 / 3.0) * (1.0 - r**2),
        np.full_like(r, 0.4 * load ** (2.0 / 3.0)),
    ])
    solution = solve_bvp(rhs, boundary, r, guess, tol=1e-8, max_nodes=500000)
    if solution.status != 0:
        raise RuntimeError(f"Hencky BVP did not converge: {solution.message}")
    return {
        "solution": solution,
        "coefficient": float(solution.sol(inner)[1] / load ** (1.0 / 3.0)),
        "n_r_centre": float(solution.sol(inner)[2] / load ** (2.0 / 3.0)),
        "n_r_edge": float(solution.sol(1.0)[2] / load ** (2.0 / 3.0)),
    }


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


def case_balloon():
    radius = 1.0
    coefficient = (1.0 - POISSON) / (MODULUS * THICKNESS)

    # Pressure sweep on ONE solver build: the pressure column is a parameter.
    nodes, triangles = icosphere(2, radius)
    pressures = np.linspace(2000.0, 20000.0, 10)
    model, solver = inflated_model(nodes, triangles, pressures[0])
    pressure_set = model.element_set("pressure")
    stretches, state = [], None
    started = time.perf_counter()
    for pressure in pressures:
        ramped = model.replaced(
            pressure_set.with_param_column("pressure", np.full(len(triangles), pressure))
        )
        solution = solver.solve(np.zeros_like(nodes), state=state, model=ramped)
        state = solution.state
        positions = state.positions
        stretches.append(np.linalg.norm(positions - positions.mean(axis=0), axis=1).mean() / radius)
        print(f"  p = {pressure:8.0f} Pa: lambda {stretches[-1]:.5f}, "
              f"{solution.iterations} it, converged={solution.converged}")
    sweep_time = time.perf_counter() - started
    exact_sweep = svk_sphere_stretch(pressures * radius * coefficient)
    print(f"  sweep of {len(pressures)} pressures on one build: {sweep_time:.2f} s, "
          f"worst relative error {np.abs(np.array(stretches) / exact_sweep - 1).max():.2e}")

    # Mesh convergence at one pressure.
    pressure = 8500.0
    exact = svk_sphere_stretch(pressure * radius * coefficient)
    sizes, errors = [], []
    for level in (1, 2, 3, 4):
        nodes, triangles = icosphere(level, radius)
        started = time.perf_counter()
        _, solver = inflated_model(nodes, triangles, pressure)
        solution = solver.solve(np.zeros_like(nodes))
        elapsed = time.perf_counter() - started
        positions = solution.state.positions
        stretch = np.linalg.norm(positions - positions.mean(axis=0), axis=1).mean() / radius
        sizes.append(mean_edge_length(nodes, triangles) / radius)
        errors.append(abs(stretch / exact - 1.0))
        print(f"  level {level}: {len(triangles)} triangles, h/R {sizes[-1]:.4f}, "
              f"relative error {errors[-1]:.3e}, {solution.iterations} it, {elapsed:.2f} s, "
              f"converged={solution.converged}")
    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    print(f"  error ratios per refinement: {[f'{r:.2f}' for r in ratios]} (4 is O(h^2))")

    figure, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    figure.suptitle(
        "SVK balloon under uniform pressure: one solver build per sweep, "
        "O(h$^2$) onto the closed form",
        fontsize=12,
    )
    fine = np.linspace(0.0, pressures[-1], 200)
    left.plot(fine / 1e3, svk_sphere_stretch(fine * radius * coefficient),
              color=COLOURS["reference"], label=r"closed form $\lambda^2 - a\lambda - 1 = 0$")
    left.plot(pressures / 1e3, stretches, "o", color=COLOURS["billow"],
              label="Billow, icosphere level 2 (320 triangles)")
    left.set_xlabel("pressure (kPa)")
    left.set_ylabel(r"stretch $\lambda$")
    left.legend(frameon=False)

    right.loglog(sizes, errors, "o-", color=COLOURS["billow"], label="Billow")
    guide = errors[0] * (np.array(sizes) / sizes[0]) ** 2
    right.loglog(sizes, guide, "--", color="0.5", label=r"$O(h^2)$")
    right.set_xlabel("mean edge length / R")
    right.set_ylabel(r"relative error in $\lambda$")
    right.legend(frameon=False)
    right.grid(alpha=0.3, which="both")
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    return figure


def case_hencky():
    radius = 1.0
    load = 1e-4
    reference = hencky_bvp(load)
    print(f"  FvK BVP at nu = {POISSON}: w0/a = {reference['coefficient']:.5f} (pa/Eh)^(1/3), "
          f"N_r centre {reference['n_r_centre']:.4f}, edge {reference['n_r_edge']:.4f} "
          f"x Eh (pa/Eh)^(2/3)")

    # Load range at one mesh: where the small-strain theory stops applying.
    ratios = np.logspace(-5, -2, 7)
    coefficients = []
    nodes, triangles, rim = disk_mesh(16, radius)
    for ratio in ratios:
        pressure = ratio * MODULUS * THICKNESS / radius
        _, solver = inflated_model(nodes, triangles, pressure, pinned=rim, closed=False)
        solution = solver.solve(np.zeros_like(nodes))
        coefficients.append(solution.state.positions[0, 2] / radius / ratio ** (1.0 / 3.0))
        print(f"  pa/Eh = {ratio:.0e}: w0/a / (pa/Eh)^(1/3) = {coefficients[-1]:.5f}, "
              f"{solution.iterations} it, converged={solution.converged}")

    # Refinement at pa/Eh = 1e-4.
    pressure = load * MODULUS * THICKNESS / radius
    rings = (8, 16, 24)
    refined, finest = [], None
    for n_rings in rings:
        nodes, triangles, rim = disk_mesh(n_rings, radius)
        started = time.perf_counter()
        model, solver = inflated_model(nodes, triangles, pressure, pinned=rim, closed=False)
        solution = solver.solve(np.zeros_like(nodes))
        elapsed = time.perf_counter() - started
        refined.append(solution.state.positions[0, 2] / radius / load ** (1.0 / 3.0))
        finest = (nodes, triangles, model, solution)
        print(f"  {n_rings} rings ({len(triangles)} triangles): coefficient {refined[-1]:.5f}, "
              f"relative error {refined[-1] / reference['coefficient'] - 1:+.2e}, "
              f"{solution.iterations} it, {elapsed:.2f} s, converged={solution.converged}")

    # Radial stress resultant from the membrane's own Green strains. The
    # principal strains are radial (larger) and hoop; N_r = Eh/(1-nu^2) (e_r + nu e_t).
    nodes, triangles, model, solution = finest
    strains = membrane_regimes(solution.state.positions, model.element_set("canopy"))
    n_r = (MODULUS * THICKNESS / (1.0 - POISSON**2)) * (
        strains["principal_1"] + POISSON * strains["principal_2"]
    )
    n_r /= MODULUS * THICKNESS * load ** (2.0 / 3.0)
    centroid_r = np.linalg.norm(nodes[triangles].mean(axis=1)[:, :2], axis=1) / radius
    # Compare each band with the BVP evaluated at the band's own mean radius:
    # the outer band of centroids sits at r ~ 0.965, not at the rim.
    inner_band = centroid_r < 1.5 / rings[-1]
    outer_band = centroid_r > 1.0 - 1.5 / rings[-1]
    inner = n_r[inner_band].mean()
    outer = n_r[outer_band].mean()
    bvp_at = lambda r: float(reference["solution"].sol(r)[2] / load ** (2.0 / 3.0))
    print(f"  N_r from Green strains at {rings[-1]} rings: centre {inner:.4f} "
          f"(BVP {bvp_at(centroid_r[inner_band].mean()):.4f} at the band radius, "
          f"{reference['n_r_centre']:.4f} at r = 0), "
          f"outer band {outer:.4f} (BVP {bvp_at(centroid_r[outer_band].mean()):.4f} at "
          f"r = {centroid_r[outer_band].mean():.3f}, {reference['n_r_edge']:.4f} at the rim)")

    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))
    figure.suptitle(
        "Hencky problem: clamped disk under uniform pressure against the FvK "
        "boundary-value solution",
        fontsize=12,
    )
    axes[0].semilogx(ratios, coefficients, "o-", color=COLOURS["billow"], label="Billow, 16 rings")
    axes[0].axhline(reference["coefficient"], color=COLOURS["reference"],
                    label=f"FvK BVP, {reference['coefficient']:.5f}")
    axes[0].axhline(0.662, color="0.6", linestyle=":", label="series value 0.662")
    axes[0].set_xlabel(r"$pa/Eh$")
    axes[0].set_ylabel(r"$w_0/a \;/\; (pa/Eh)^{1/3}$")
    axes[0].legend(frameon=False)

    axes[1].plot(rings, refined, "o-", color=COLOURS["billow"], label="Billow")
    axes[1].axhline(reference["coefficient"], color=COLOURS["reference"], label="FvK BVP")
    axes[1].set_xlabel("rings")
    axes[1].set_ylabel(r"$w_0/a \;/\; (pa/Eh)^{1/3}$ at $pa/Eh = 10^{-4}$")
    axes[1].legend(frameon=False)

    profile = np.linspace(1e-4, 1.0, 300)
    axes[2].plot(profile, reference["solution"].sol(profile)[2] / load ** (2.0 / 3.0),
                 color=COLOURS["reference"], label="FvK BVP")
    axes[2].plot(centroid_r, n_r, ".", color=COLOURS["billow"], markersize=3,
                 label=f"Billow, {rings[-1]} rings, per triangle")
    axes[2].set_xlabel("r / a")
    axes[2].set_ylabel(r"$N_r \;/\; Eh\,(pa/Eh)^{2/3}$")
    axes[2].legend(frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    return figure


def case_cylinder():
    radius, pressure = 1.0, 1500.0
    exact = np.sqrt(1.0 + 2.0 * pressure * radius * (1.0 - POISSON**2) / (MODULUS * THICKNESS))
    print(f"  plane-strain hoop stretch: {exact:.5f}")

    figure, axes = plt.subplots(1, 1, figsize=(7.0, 4.0))
    figure.suptitle(
        "Open cylinder between rigid end rings: hoop stretch against the "
        "plane-strain closed form",
        fontsize=12,
    )
    for aspect, colour in ((4.0, PALETTE["Sky Blue"]), (8.0, COLOURS["billow"])):
        nodes, triangles, rings, stations = cylinder_mesh(radius, aspect * radius, 48)
        started = time.perf_counter()
        _, solver = inflated_model(nodes, triangles, pressure, pinned=rings, closed=False)
        solution = solver.solve(np.zeros_like(nodes))
        elapsed = time.perf_counter() - started
        positions = solution.state.positions.reshape(len(stations), -1, 3)
        hoop = np.linalg.norm(positions[:, :, :2], axis=2).mean(axis=1) / radius
        middle = hoop[len(stations) // 2]
        settled = np.flatnonzero(np.abs(hoop / exact - 1.0) < 1e-3)
        layer = stations[settled[0]] / radius if settled.size else np.nan
        print(f"  L/R = {aspect:.0f}: {len(triangles)} triangles, mid-length stretch {middle:.5f} "
              f"(error {middle / exact - 1:+.2e}), boundary layer ~{layer:.2f} R, "
              f"{solution.iterations} it, {elapsed:.2f} s, converged={solution.converged}")
        axes.plot(stations / radius, hoop, "-", color=colour, label=f"Billow, L/R = {aspect:.0f}")
    axes.axhline(exact, color=COLOURS["reference"], linestyle="--",
                 label=r"plane strain, $\lambda_\theta^2 - 1 = 2pR(1-\nu^2)/Et$")
    axes.set_xlabel("z / R")
    axes.set_ylabel(r"hoop stretch $\lambda_\theta$")
    axes.legend(frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    return figure


CASES = {
    "balloon": case_balloon,
    "hencky": case_hencky,
    "cylinder": case_cylinder,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=sorted(CASES), action="append")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no-title", action="store_true",
                        help="drop figure titles (for a paper, where the caption carries them)")
    arguments = parser.parse_args()
    if arguments.no_title:
        import matplotlib.figure, matplotlib.axes
        matplotlib.figure.Figure.suptitle = lambda self, *a, **k: None

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    for name in arguments.case or sorted(CASES):
        print(f"[{name}]")
        started = time.perf_counter()
        figure = CASES[name]()
        target = arguments.output_dir / f"pressure_{name}.png"
        figure.savefig(target, dpi=160)
        print(f"  -> {target}  ({time.perf_counter() - started:.1f} s)")
        if not arguments.show:
            plt.close(figure)
    if arguments.show:
        plt.show()


if __name__ == "__main__":
    main()
