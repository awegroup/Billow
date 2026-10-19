# Copyright (c) 2023-2026 Oriol Cayon
#
# SPDX-License-Identifier: Apache-2.0

"""Uniform pressure as a potential: the ``-pV`` kernel and the surfaces it loads.

Three layers of check, each against something exact:

* discrete identities -- the enclosed volume of a cube, the kernel's gradient
  against the lumped ``p A n / 3`` force, zero resultant and moment on a closed
  surface, the origin dependence confined to the rim of an open one;
* exact closed forms -- a vertex-transitive polyhedron inflates by a uniform
  scaling whose factor is the root of one quadratic, matched to 1e-6;
* convergence -- an icosphere onto the continuum SVK balloon at O(h^2), and a
  pinned disk onto the Foppl-von Karman (Hencky) membrane solution.

The pressure tolerance is scaled by hand in every solve: the solver scales its
acceptance with the EXTERNAL forces only, and here the load is inside the
energy.
"""

from __future__ import annotations

import numpy as np
import pytest

from billow import MinimumEnergySolver, StructuralModel
from billow.elements import (
    build_membrane_elements,
    build_pressure_elements,
    check_closed_surface,
    enclosed_volume,
    lumped_pressure_forces,
    pressure_forces,
    surface_boundary_nodes,
)
from billow.energy import PotentialEnergy

MODULUS = 5.0e8  # Pa
POISSON = 0.3
THICKNESS = 3.0e-4  # m


# --------------------------------------------------------------------------
# Meshes -- every one wound outward
# --------------------------------------------------------------------------


def wound_outward(nodes: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Re-wind each triangle of a star-shaped surface about the origin to face out."""
    triangles = np.array(triangles, dtype=int)
    for k, (a, b, c) in enumerate(triangles):
        if np.dot(nodes[a], np.cross(nodes[b], nodes[c])) < 0.0:
            triangles[k] = [a, c, b]
    return triangles


def polyhedron(name: str, radius: float = 1.0):
    """Vertex-transitive equilateral polyhedra with circumradius ``radius``."""
    if name == "tetrahedron":
        nodes = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
        faces = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
    elif name == "octahedron":
        nodes = np.vstack([np.eye(3), -np.eye(3)])
        faces = [[x, y, z] for x in (0, 3) for y in (1, 4) for z in (2, 5)]
    elif name == "icosahedron":
        phi = 0.5 * (1.0 + np.sqrt(5.0))
        nodes = np.array(
            [[-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
             [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
             [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]], float,
        )
        faces = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
                 [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                 [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
                 [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]]
    else:
        raise ValueError(name)
    nodes *= radius / np.linalg.norm(nodes[0])
    return nodes, wound_outward(nodes, np.array(faces))


def icosphere(level: int, radius: float = 1.0):
    """Icosahedron subdivided ``level`` times, every node projected to the sphere."""
    nodes, triangles = polyhedron("icosahedron", radius)
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


def box_cells(n_cells: int):
    """``n_cells`` unit cubes in a row; returns the nodes and one outward-wound
    closed triangle list per cell, so neighbouring cells share their wall."""
    nodes = np.array(
        [[x, y, z] for x in range(n_cells + 1) for y in (0, 1) for z in (0, 1)], float
    )

    def index(x, y, z):
        return 4 * x + 2 * y + z

    cells = []
    for x in range(n_cells):
        quads = [
            [index(x, 0, 0), index(x, 1, 0), index(x + 1, 1, 0), index(x + 1, 0, 0)],
            [index(x, 0, 1), index(x + 1, 0, 1), index(x + 1, 1, 1), index(x, 1, 1)],
            [index(x, 0, 0), index(x + 1, 0, 0), index(x + 1, 0, 1), index(x, 0, 1)],
            [index(x, 1, 0), index(x, 1, 1), index(x + 1, 1, 1), index(x + 1, 1, 0)],
            [index(x, 0, 0), index(x, 0, 1), index(x, 1, 1), index(x, 1, 0)],
            [index(x + 1, 0, 0), index(x + 1, 1, 0), index(x + 1, 1, 1), index(x + 1, 0, 1)],
        ]
        triangles = []
        for a, b, c, d in quads:
            triangles += [[a, b, c], [a, c, d]]
        cells.append(np.array(triangles))
    return nodes, cells


def disk_mesh(n_rings: int, radius: float = 1.0):
    """Flat disk of ``6 k`` nodes on ring ``k``, triangles wound to face ``+z``.

    Returns ``(nodes, triangles, rim)`` with the centre at node 0.
    """
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


def surface_area(nodes: np.ndarray, triangles: np.ndarray) -> float:
    corners = nodes[triangles]
    return float(
        0.5 * np.linalg.norm(
            np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1
        ).sum()
    )


def pressure_tolerance(pressure: float, area: float, n_nodes: int) -> float:
    """``force_tolerance`` for a pressure-only solve: 1e-5 of a nodal pressure force.

    Pressure is not part of the solver's ``load_scale`` (it lives inside the
    energy), so the absolute default of 1e-6 N would be meaningless against
    nodal pressure forces of order ``p A / n``. 1e-5 of that force keeps the
    verdict on the residual rather than on IPOPT's own status, including on the
    finest Hencky disk (residual 3e-8 N against a 2.6e-7 N tolerance).
    """
    return 1e-5 * pressure * area / n_nodes


def svk_sphere_stretch(coefficient: float) -> float:
    """Root of ``lambda^2 - a lambda - 1 = 0``: the inflated SVK sphere.

    Per reference area the equibiaxial SVK energy is ``t E e^2 / (1 - nu)`` with
    ``e = (lambda^2 - 1) / 2``; stationarity of ``Pi(lambda)`` against the
    ``-pV`` term gives the quadratic with ``a = p R0 (1 - nu) / (E t)`` for the
    continuum sphere and ``a = 3 p V0 (1 - nu) / (A0 E t)`` for a
    vertex-transitive polyhedron, where uniform scaling is the symmetric
    critical point.
    """
    return 0.5 * (coefficient + np.sqrt(coefficient**2 + 4.0))


def inflated_surface(nodes, triangles, pressure, *, pinned=(), closed=True, **kwargs):
    """Fabric plus pressure over one surface; returns ``(solution, model)``."""
    fabric = build_membrane_elements(nodes, triangles, THICKNESS, MODULUS, POISSON, **kwargs)
    pressure_set = build_pressure_elements(nodes, triangles, pressure, closed=closed)
    model = StructuralModel(
        nodes, [fabric, pressure_set], fixed_translation_nodes=list(pinned)
    )
    tolerance = pressure_tolerance(pressure, surface_area(nodes, triangles), len(nodes))
    solver = MinimumEnergySolver(model, tolerance=1e-10, force_tolerance=tolerance)
    return solver.solve(np.zeros_like(nodes)), model


# --------------------------------------------------------------------------
# Discrete identities
# --------------------------------------------------------------------------


def test_enclosed_volume_is_exact_on_a_cube_and_a_tetrahedron():
    nodes, (cube,) = box_cells(1)
    assert enclosed_volume(nodes, cube) == pytest.approx(1.0, abs=1e-14)

    corner = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    # Wind about an interior point (one vertex sits on the origin); the volume
    # of the closed surface itself does not care where the origin is.
    faces = wound_outward(corner - 0.1, [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    assert enclosed_volume(corner, faces) == pytest.approx(1.0 / 6.0, abs=1e-14)


def test_icosphere_volume_converges_at_second_order():
    errors = []
    for level in (1, 2, 3):
        nodes, triangles = icosphere(level)
        errors.append(abs(enclosed_volume(nodes, triangles) / (4.0 / 3.0 * np.pi) - 1.0))
    for coarse, fine in zip(errors[:-1], errors[1:]):
        assert coarse / fine > 3.4, errors


def test_mapped_energy_is_minus_pressure_times_volume():
    """The mapped kernel sums to exactly ``-pV`` of the whole closed surface."""
    nodes, triangles = icosphere(1)
    pressure = 250.0
    model = StructuralModel(nodes, [build_pressure_elements(nodes, triangles, pressure)])
    energy = PotentialEnergy(model, anchor_stiffness=0.0)

    positions = nodes + np.random.default_rng(3).normal(scale=0.05, size=nodes.shape)
    parameters = energy.pack_parameters(
        model.initial_state(), model.element_sets, np.zeros_like(positions), None, positions
    )
    total = float(energy.total_energy(energy.pack_unknowns(positions), parameters))
    assert total == pytest.approx(-pressure * enclosed_volume(positions, triangles), rel=1e-12)


def test_closed_surface_check_rejects_open_flipped_inward_and_shared_rib_meshes():
    nodes, (cube,) = box_cells(1)
    check_closed_surface(nodes, cube)

    disk_nodes, disk, _ = disk_mesh(2)
    with pytest.raises(ValueError, match="open"):
        check_closed_surface(disk_nodes, disk)

    flipped = cube.copy()
    flipped[0] = flipped[0, [0, 2, 1]]
    with pytest.raises(ValueError, match="inconsistently wound"):
        check_closed_surface(nodes, flipped)

    with pytest.raises(ValueError, match="wound inward"):
        check_closed_surface(nodes, cube[:, [0, 2, 1]])

    # A mirror-symmetric surface whose far half was copied WITHOUT reversing
    # the winding: the triple product is odd under a reflection, so that half
    # faces inward and meets the near half edge-on in the mirror plane.
    octa_nodes, octa = polyhedron("octahedron")
    near = octa[np.all(octa_nodes[octa][:, :, 0] >= 0.0, axis=1)]
    mirror = {0: 3, 3: 0}
    far = np.vectorize(lambda node: mirror.get(node, node))(near)
    with pytest.raises(ValueError, match="inconsistently wound"):
        check_closed_surface(octa_nodes, np.vstack([near, far]))

    two_nodes, (left, right) = box_cells(2)
    with pytest.raises(ValueError, match="non-manifold"):
        check_closed_surface(two_nodes, np.vstack([left, right]))
    with pytest.raises(ValueError, match="non-manifold"):
        build_pressure_elements(two_nodes, np.vstack([left, right]), 100.0)


def test_builder_rejects_a_per_triangle_pressure_and_diagnostic_rejects_a_nonuniform_column():
    nodes, (cube,) = box_cells(1)
    with pytest.raises(ValueError, match="scalar"):
        build_pressure_elements(nodes, cube, np.full(len(cube), 100.0))
    # A NumPy scalar is a scalar.
    build_pressure_elements(nodes, cube, np.float64(100.0))

    pressure_set = build_pressure_elements(nodes, cube, 100.0)
    skewed = pressure_set.with_param_column("pressure", np.linspace(90.0, 110.0, len(cube)))
    with pytest.raises(ValueError, match="uniform"):
        pressure_forces(nodes, skewed)

    # Two separate surfaces in one set may sit at different pressures.
    two_nodes, (left, right) = box_cells(2)
    nodes_apart = np.vstack([two_nodes[:8], two_nodes[:8] + [5.0, 0.0, 0.0]])
    two = build_pressure_elements(nodes_apart, np.vstack([left, left + 8]), 100.0)
    two = two.with_param_column("pressure", np.repeat([100.0, 200.0], 12))
    pressure_forces(nodes_apart, two)


def test_pressure_forces_match_finite_differences():
    nodes, triangles = icosphere(1)
    pressure = 300.0
    pressure_set = build_pressure_elements(nodes, triangles, pressure)
    rng = np.random.default_rng(5)
    positions = nodes + rng.normal(scale=0.1, size=nodes.shape)

    analytic = pressure_forces(positions, pressure_set)
    step = 1e-6
    for node in rng.choice(len(nodes), size=6, replace=False):
        for axis in range(3):
            forward, backward = positions.copy(), positions.copy()
            forward[node, axis] += step
            backward[node, axis] -= step
            gradient = -pressure * (
                enclosed_volume(forward, triangles) - enclosed_volume(backward, triangles)
            ) / (2.0 * step)
            assert analytic[node, axis] == pytest.approx(-gradient, rel=1e-6, abs=1e-6)


def test_pressure_forces_equal_lumped_pressure_forces_on_a_closed_surface():
    nodes, triangles = icosphere(2)
    pressure = 120.0
    positions = nodes + np.random.default_rng(1).normal(scale=0.05, size=nodes.shape)
    from_kernel = pressure_forces(positions, build_pressure_elements(nodes, triangles, pressure))
    lumped = lumped_pressure_forces(positions, triangles, pressure)
    scale = np.linalg.norm(lumped, axis=1).max()
    assert np.abs(from_kernel - lumped).max() < 1e-12 * scale


def test_net_force_and_moment_vanish_on_a_closed_surface():
    nodes, triangles = icosphere(2)
    pressure = 120.0
    positions = nodes + np.random.default_rng(2).normal(scale=0.05, size=nodes.shape)
    forces = pressure_forces(positions, build_pressure_elements(nodes, triangles, pressure))
    scale = pressure * surface_area(positions, triangles)
    assert np.abs(forces.sum(axis=0)).max() < 1e-12 * scale
    assert np.abs(np.cross(positions, forces).sum(axis=0)).max() < 1e-12 * scale


def test_open_surface_rim_forces_depend_on_the_origin():
    """Only the rim of an open surface sees where the origin is.

    An interior node's cone terms close into a ring polygon whose vector area
    is translation-invariant; a rim node's do not. That is why an open
    pressure surface is a potential exactly when its rim is pinned.
    """
    nodes, triangles, rim = disk_mesh(4)
    pressure_set = build_pressure_elements(nodes, triangles, 100.0, closed=False)
    assert np.array_equal(surface_boundary_nodes(triangles), rim)

    positions = nodes + np.random.default_rng(0).normal(scale=0.02, size=nodes.shape)
    shifted = positions + np.array([3.0, -2.0, 7.0])
    change = pressure_forces(shifted, pressure_set) - pressure_forces(positions, pressure_set)
    interior = np.setdiff1d(np.arange(len(nodes)), rim)
    scale = 100.0 * np.pi / len(nodes)
    assert np.abs(change[interior]).max() < 1e-12 * scale
    assert np.abs(change[rim]).max() > 0.1 * scale


# --------------------------------------------------------------------------
# Exact closed forms and convergence
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["tetrahedron", "octahedron", "icosahedron"])
def test_vertex_transitive_polyhedron_inflates_by_uniform_scaling(name):
    """Free balloon, no pinned node: the symmetric critical point is exact.

    Rigid modes are held only by the weak anchor, so the body drifts by a
    fraction of a micron and the radii are measured about the centroid.
    """
    nodes, triangles = polyhedron(name)
    pressure = 8500.0
    solution, _ = inflated_surface(nodes, triangles, pressure)
    assert solution.converged, solution.status

    area, volume = surface_area(nodes, triangles), enclosed_volume(nodes, triangles)
    stretch = svk_sphere_stretch(
        3.0 * pressure * volume * (1.0 - POISSON) / (area * MODULUS * THICKNESS)
    )
    positions = solution.state.positions
    radii = np.linalg.norm(positions - positions.mean(axis=0), axis=1)
    assert radii == pytest.approx(stretch, rel=1e-6)


def test_balloon_converges_to_the_svk_sphere_at_second_order():
    """Icosphere levels 1-3 against the continuum SVK sphere, ``a ~ 0.04``."""
    pressure = 8500.0
    stretch = svk_sphere_stretch(pressure * (1.0 - POISSON) / (MODULUS * THICKNESS))
    errors = []
    for level in (1, 2, 3):
        nodes, triangles = icosphere(level)
        solution, _ = inflated_surface(nodes, triangles, pressure)
        assert solution.converged, f"{level=} {solution.status}"
        positions = solution.state.positions
        radius = np.linalg.norm(positions - positions.mean(axis=0), axis=1).mean()
        errors.append(abs(radius / stretch - 1.0))
    assert errors[-1] < 1e-4, errors
    for coarse, fine in zip(errors[:-1], errors[1:]):
        assert coarse / fine > 3.4, errors


def test_hencky_disk_converges_onto_the_fvk_solution():
    """Clamped circular membrane under uniform pressure, the Hencky problem.

    Reference: the Foppl-von Karman clamped membrane at ``nu = 0.3`` solved as
    a boundary-value problem with ``scipy.integrate.solve_bvp``
    (``validation/run_pressure_benchmarks.py``), which gives
    ``w0 / a = 0.65344 (p a / E h)^(1/3)``. The textbook 0.662 is the
    uncorrected series value. Run at ``p a / E h = 1e-4``, where finite-strain
    effects sit well below the discretisation error; the CST disk converges
    onto the coefficient from above.
    """
    ratio = 1e-4
    pressure = ratio * MODULUS * THICKNESS
    errors = []
    for n_rings in (8, 16, 24):
        nodes, triangles, rim = disk_mesh(n_rings)
        solution, _ = inflated_surface(nodes, triangles, pressure, pinned=rim, closed=False)
        assert solution.converged, f"{n_rings=} {solution.status}"
        coefficient = solution.state.positions[0, 2] / ratio ** (1.0 / 3.0)
        errors.append(coefficient / 0.65344 - 1.0)
    assert all(error > 0.0 for error in errors), errors
    assert errors[0] > errors[1] > errors[2], errors
    assert errors[-1] < 1e-3, errors

    # The clamped disk is biaxially taut everywhere, so the relaxed and the
    # plain SVK law must agree.
    nodes, triangles, rim = disk_mesh(8)
    unrelaxed, _ = inflated_surface(
        nodes, triangles, pressure, pinned=rim, closed=False, wrinkling=False
    )
    relaxed, _ = inflated_surface(nodes, triangles, pressure, pinned=rim, closed=False)
    assert unrelaxed.state.positions[0, 2] == pytest.approx(
        relaxed.state.positions[0, 2], rel=1e-4
    )


def test_lagged_dead_load_converges_to_the_pressure_potential():
    """The adapter's route for a non-uniform field, checked on a uniform one.

    ``lumped_pressure_forces`` evaluated on the previous shape and applied as a
    dead load, iterated to a fixed point: there the lumped force on each
    interior node equals the gradient of ``-pV`` (uniform ``p``, closed fan),
    so the fixed point is the potential's own equilibrium.
    """
    ratio = 1e-4
    pressure = ratio * MODULUS * THICKNESS
    nodes, triangles, rim = disk_mesh(8)
    potential, _ = inflated_surface(nodes, triangles, pressure, pinned=rim, closed=False)

    fabric = build_membrane_elements(nodes, triangles, THICKNESS, MODULUS, POISSON)
    model = StructuralModel(nodes, [fabric], fixed_translation_nodes=rim)
    solver = MinimumEnergySolver(model, tolerance=1e-10)
    positions, state = nodes.copy(), None
    for _ in range(20):
        solution = solver.solve(lumped_pressure_forces(positions, triangles, pressure), state=state)
        assert solution.converged, solution.status
        change = np.abs(solution.state.positions - positions).max()
        positions, state = solution.state.positions, solution.state
        if change < 1e-9:
            break
    assert change < 1e-9, "the lagged iteration did not settle"
    assert positions[0, 2] == pytest.approx(potential.state.positions[0, 2], rel=1e-6)


def test_pressure_ramp_reuses_one_solver():
    """``with_param_column("pressure")`` + ``model.replaced`` ramps on one build."""
    nodes, triangles, rim = disk_mesh(6)
    fabric = build_membrane_elements(nodes, triangles, THICKNESS, MODULUS, POISSON)
    pressure_set = build_pressure_elements(nodes, triangles, 1.0, closed=False)
    model = StructuralModel(nodes, [fabric, pressure_set], fixed_translation_nodes=rim)
    pressures = [5.0, 15.0, 45.0]
    tolerance = pressure_tolerance(pressures[0], np.pi, len(nodes))

    solver = MinimumEnergySolver(model, tolerance=1e-10, force_tolerance=tolerance)
    volumes, state = [], None
    for pressure in pressures:
        ramped = model.replaced(
            pressure_set.with_param_column("pressure", np.full(len(triangles), pressure))
        )
        solution = solver.solve(np.zeros_like(nodes), state=state, model=ramped)
        assert solution.converged, solution.status
        state = solution.state
        volumes.append(enclosed_volume(state.positions, triangles))
    assert volumes[0] < volumes[1] < volumes[2]

    cold, _ = inflated_surface(nodes, triangles, pressures[-1], pinned=rim, closed=False)
    assert np.abs(state.positions - cold.state.positions).max() < 1e-6
