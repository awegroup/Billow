# Copyright (c) 2023-2026 Oriol Cayon
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Uniform gauge pressure on a triangulated membrane surface, as a potential.

A ram-air cell, an inflatable tube modelled with its own fabric, or a single
skin pinned along its edge under a uniform test pressure is loaded by ``p`` inside
the same energy minimisation as the fabric that carries it. The work of a
uniform pressure is ``p`` times the swept volume, so its potential is

    U = -p V,    V = sum_t x1 . (x2 x x3) / 6,

the enclosed volume written as a sum of signed cone volumes from the origin,
one per triangle. Each cone is a translation-only three-node element, mapped
over the surface like any other kernel. ``-dU/dx_i = p dV/dx_i``, and for a
node whose fan of triangles is closed ``dV/dx_i`` is the vector area of the
ring polygon around it, ``(1/3) sum_t A_t n_t``: the pressure force a lumped
``p A n / 3`` scheme would apply, exactly, at every interior node. Positive
``p`` inflates when the triangles are wound OUTWARD (counter-clockwise seen
from outside); :func:`build_pressure_elements` checks the sign of the total
volume and refuses an inward mesh.

Closed and open surfaces
------------------------
The cone volumes depend on where the origin is; their sum over a closed
surface does not, and neither does the gradient at any node whose fan is
closed. An OPEN surface (a canopy pinned along its edge, a cylinder between
two rigid rings) is therefore also a genuine potential, provided every rim
node is pinned: only the rim nodes carry origin-dependent cone terms, and a
pinned node's gradient never enters the equilibrium. ``closed=False`` relaxes
the closure check for that case; the caller pins
:func:`surface_boundary_nodes`.

One scalar per set
------------------
The exactness at interior nodes holds only for a pressure that is UNIFORM over
the node's fan. With a per-triangle ``p_t`` the sum ``-sum_t p_t V_t`` is no
longer translation-invariant (it changes by ``20 N`` on a disk translated a
few metres, against ``1e-13 N`` for uniform ``p``), so the gradient is not a
pressure force at all. ``pressure`` therefore stays a parameter column, so a
ramp runs on one solver build through ``ElementSet.with_param_column``, but
it is ONE scalar per element set: the builder fills the column uniformly and
refuses an array, and :func:`pressure_forces` refuses a column that is not
constant over each connected piece of the surface. Callers that write the
column themselves must keep it that way.

Several cells at different pressures are several sets, one closed surface
each. A rib shared by two cells appears in both sets with opposite windings
and carries the net ``p_left - p_right``; at equal pressure an internal rib
carries nothing and belongs in no pressure set at all (a single set holding
two cells that share a rib is rejected as non-manifold). A mirror-symmetric
canopy needs its mirrored half wound in REVERSED order: the triple product is
odd under a reflection, and the orientation check catches a half that was
copied without the flip.

What this kernel is not
-----------------------
A flow-dependent pressure distribution -- the aerodynamic ``dp`` on a single
skin, a stagnation-pressure gradient -- is non-uniform and not a potential.
That load goes through :func:`lumped_pressure_forces`, a NumPy helper that
lumps ``p_t A_t n_t / 3`` at the nodes, applied by the adapter as a dead load
lagged by one coupling step, in the same way as drag in
``examples/run_quasi_steady_tether.py``. For uniform ``p`` on a closed fan the
two routes agree exactly at the fixed point. A fixed-mass gas law
(``-p0 V0 ln(V/V0)``) is a function of the global volume, not of one element,
and is out of scope; an open ram inlet is a free boundary and not
conservative.

Two things to carry into a solve:

* :attr:`billow.StructuralSolution.strain_energy` is the TOTAL element energy
  and includes ``-pV``, so on an inflated structure it is negative and large.
  The elastic part is the membrane set's own energy.
* The solver's acceptance tolerance scales with the EXTERNAL forces only.
  Pressure lives inside the energy, so a pressure-only solve is accepted at
  the absolute ``force_tolerance`` (1e-6 N by default) against nodal pressure
  forces of order ``p A / n_nodes``; pass a ``force_tolerance`` scaled to that,
  especially with ``move_limit`` set, where the residual alone decides.

The SVK membrane and its relaxed form both grow like ``lambda^4`` in a uniform
stretch while ``pV`` grows like ``lambda^3``, so the total energy of an
inflated balloon is bounded below and a one-shot solve is well posed; the
limit-point instability of a neo-Hookean balloon is real physics this
constitutive law cannot show.
"""

from __future__ import annotations

from dataclasses import dataclass

import casadi as ca
import numpy as np

from .base import ElementSet, element_translations

Array = np.ndarray

PARAM_NAMES: tuple[str, ...] = ("pressure",)


def signed_cone_volume(x1, x2, x3, xp):
    """Signed volume ``x1 . (x2 x x3) / 6`` of the cone from the origin over a triangle.

    The single place the formula is written: the CasADi kernel and every NumPy
    diagnostic call it. With ``xp is np`` the inputs are ``(n, 3)`` rows and
    the result is ``(n,)``.
    """
    if xp is np:
        x1, x2, x3 = (np.atleast_2d(np.asarray(x, dtype=float)) for x in (x1, x2, x3))
        return np.einsum("ij,ij->i", x1, np.cross(x2, x3)) / 6.0
    return xp.dot(x1, xp.cross(x2, x3)) / 6.0


@dataclass(frozen=True)
class PressureKernel:
    """Potential ``-p V_cone`` of one outward-wound surface triangle."""

    nodes_per_element: int = 3
    rotational_nodes: tuple[int, ...] = ()
    param_names: tuple[str, ...] = PARAM_NAMES

    def energy(self, q, frames, p):
        return -p[0] * signed_cone_volume(
            element_translations(q, 0),
            element_translations(q, 1),
            element_translations(q, 2),
            ca,
        )


# ---------------------------------------------------------------------------
# Surface topology
# ---------------------------------------------------------------------------


def _directed_edges(triangles: Array) -> Array:
    triangles = np.atleast_2d(np.asarray(triangles, dtype=int))
    return np.concatenate(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]
    )


def surface_edges(triangles: Array) -> tuple[Array, Array, Array]:
    """Classify the edges of a triangulation by directed-edge counting.

    Returns ``(boundary, inconsistent, nonmanifold)``, each a ``(k, 2)`` array
    of node pairs. On a closed, consistently wound manifold surface every
    undirected edge appears exactly once in each direction, so all three are
    empty. An edge seen once is a boundary edge; one seen twice in the SAME
    direction joins two triangles wound against each other; one seen more
    than twice is shared by three or more triangles.
    """
    directed = _directed_edges(triangles)
    forward = directed[:, 0] < directed[:, 1]
    undirected = np.sort(directed, axis=1)
    edges, inverse = np.unique(undirected, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    total = np.bincount(inverse, minlength=len(edges))
    forward_count = np.bincount(inverse, weights=forward, minlength=len(edges))
    backward_count = total - forward_count

    boundary = edges[total == 1]
    nonmanifold = edges[total > 2]
    inconsistent = edges[(total == 2) & ((forward_count == 2) | (backward_count == 2))]
    return boundary, inconsistent, nonmanifold


def surface_boundary_nodes(triangles: Array) -> Array:
    """Sorted node indices on the boundary of an open triangulation.

    These are the nodes a ``closed=False`` pressure set needs pinned: their
    cone terms depend on the origin, every other node's do not.
    """
    boundary, _, _ = surface_edges(triangles)
    return np.unique(boundary)


def surface_components(triangles: Array) -> Array:
    """Connected-component label per triangle, triangles joined by shared NODES.

    Node connectivity rather than edge connectivity because the uniform-
    pressure requirement is per node fan: two patches touching at a single
    vertex share that vertex's fan.
    """
    triangles = np.atleast_2d(np.asarray(triangles, dtype=int))
    parent = np.arange(int(triangles.max()) + 1 if triangles.size else 0)

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    for a, b, c in triangles:
        root = find(int(a))
        for node in (int(b), int(c)):
            other = find(node)
            if other != root:
                parent[other] = root
    roots = np.array([find(int(a)) for a in triangles[:, 0]])
    _, labels = np.unique(roots, return_inverse=True)
    return np.asarray(labels).reshape(-1)


def enclosed_volume(positions: Array, triangles: Array) -> float:
    """Volume [m^3] enclosed by an outward-wound closed triangulation.

    On an open surface this is the cone volume from the origin, which depends
    on where the origin is; it is still the quantity the kernel integrates.
    """
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    triangles = np.atleast_2d(np.asarray(triangles, dtype=int))
    cones = signed_cone_volume(
        positions[triangles[:, 0]], positions[triangles[:, 1]], positions[triangles[:, 2]], np
    )
    return float(cones.sum())


def check_closed_surface(nodes: Array, triangles: Array) -> None:
    """Raise ``ValueError`` unless ``triangles`` is a closed, outward-wound manifold.

    The message names the first offending edges, so a mis-wound mirror half
    or a wall shared by two cells in one set can be found in the mesh.
    """
    _check_winding(triangles)
    boundary, _, _ = surface_edges(triangles)
    if len(boundary):
        raise ValueError(
            f"pressure surface is open: {len(boundary)} boundary edges, first "
            f"{[tuple(int(n) for n in edge) for edge in boundary[:4]]}; pin the "
            "boundary nodes and pass closed=False if that is intended"
        )
    volume = enclosed_volume(nodes, triangles)
    if volume <= 0.0:
        raise ValueError(
            f"pressure surface is wound inward (enclosed volume {volume:.3e} <= 0); "
            "reverse the triangle winding"
        )


def _check_winding(triangles: Array) -> None:
    _, inconsistent, nonmanifold = surface_edges(triangles)
    if len(nonmanifold):
        raise ValueError(
            f"pressure surface is non-manifold: {len(nonmanifold)} edges shared by "
            "more than two triangles, first "
            f"{[tuple(int(n) for n in edge) for edge in nonmanifold[:4]]}; two "
            "cells sharing a rib need one pressure set each"
        )
    if len(inconsistent):
        raise ValueError(
            f"pressure surface is inconsistently wound: {len(inconsistent)} edges "
            "traversed twice in the same direction, first "
            f"{[tuple(int(n) for n in edge) for edge in inconsistent[:4]]}"
        )


# ---------------------------------------------------------------------------
# Forces
# ---------------------------------------------------------------------------


def lumped_pressure_forces(positions: Array, triangles: Array, pressure) -> Array:
    """Nodal forces ``(n_nodes, 3)`` of a pressure field lumped as ``p_t A_t n_t / 3``.

    ``pressure`` is a scalar or one value per triangle, positive along the
    triangle normal (outward for an outward-wound surface). NumPy only: this
    is the route for a NON-uniform field -- an aerodynamic ``dp`` over a
    single skin -- which has no potential and is applied by the adapter as a
    dead load evaluated on the previous step's shape, like drag in
    ``examples/run_quasi_steady_tether.py``. For uniform ``p`` the lumped force
    at an interior node equals the gradient of ``-pV``, so the lagged iteration
    and :class:`PressureKernel` share their fixed point.
    """
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    triangles = np.atleast_2d(np.asarray(triangles, dtype=int))
    pressure = np.broadcast_to(np.asarray(pressure, dtype=float), (len(triangles),))

    corners = positions[triangles]
    twice_area_normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    share = pressure[:, None] * twice_area_normal / 6.0

    forces = np.zeros_like(positions)
    for corner in range(3):
        np.add.at(forces, triangles[:, corner], share)
    return forces


def _check_uniform_pressure(element_set: ElementSet) -> None:
    """The contract ``with_param_column("pressure", ...)`` callers must keep."""
    pressure = element_set.param_column("pressure")
    labels = surface_components(element_set.connectivity)
    for label in np.unique(labels):
        values = pressure[labels == label]
        if not np.allclose(values, values[0], rtol=1e-12, atol=0.0):
            raise ValueError(
                f"pressure set {element_set.name!r}: the pressure column is not "
                f"uniform over connected surface {label} (spread "
                f"{values.min():.6g} .. {values.max():.6g} Pa); a per-triangle "
                "pressure is not a potential, use lumped_pressure_forces"
            )


def pressure_forces(positions: Array, element_set: ElementSet) -> Array:
    """Nodal pressure forces ``(n_nodes, 3)``, ``-dU/dx`` from the kernel's own gradient.

    Mapped over the set like :func:`billow.elements.line_tensions`, so the
    forces are exactly what the solver balanced. Exact at every node of a
    closed surface and at every interior node of an open one; the rim values
    of an open surface depend on the origin and are only meaningful on pinned
    nodes, where they read as the reaction the pin must supply.
    """
    _check_uniform_pressure(element_set)
    kernel = element_set.kernel
    q = ca.SX.sym("q", 3 * kernel.nodes_per_element)
    p = ca.SX.sym("p", len(kernel.param_names))
    force = -ca.gradient(kernel.energy(q, ca.SX(0, 1), p), q)
    evaluate = ca.Function("pressure_force", [q, p], [force]).map(element_set.n_elements)

    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    connectivity = element_set.connectivity
    element_positions = positions[connectivity].reshape(element_set.n_elements, -1)
    element_forces = np.asarray(evaluate(element_positions.T, element_set.params.T)).T
    element_forces = element_forces.reshape(element_set.n_elements, kernel.nodes_per_element, 3)

    forces = np.zeros_like(positions)
    for corner in range(kernel.nodes_per_element):
        np.add.at(forces, connectivity[:, corner], element_forces[:, corner])
    return forces


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_pressure_elements(
    nodes: Array,
    triangles: Array,
    pressure: float,
    *,
    name: str = "pressure",
    closed: bool = True,
) -> ElementSet:
    """Assemble the ``-pV`` set over an outward-wound triangulated surface.

    ``pressure`` [Pa] is ONE scalar, gauge, positive inflating; the column is
    filled uniformly. Unlike the membrane builder's broadcast convention an
    array is refused outright (``np.ndim(pressure) > 0``), because a
    per-triangle pressure is not a potential (see the module docstring); a
    NumPy scalar passes. With ``closed`` the surface must be a closed,
    consistently wound, outward manifold; with ``closed=False`` only the
    winding is checked and the caller must pin every node of
    :func:`surface_boundary_nodes`.
    """
    if np.ndim(pressure) > 0:
        raise ValueError(
            f"pressure set {name!r}: pressure must be one scalar per set, got an "
            f"array of shape {np.shape(pressure)}; a per-triangle pressure is not "
            "a potential, use lumped_pressure_forces for a non-uniform field"
        )
    triangles = np.atleast_2d(np.asarray(triangles, dtype=int))
    if closed:
        check_closed_surface(nodes, triangles)
    else:
        _check_winding(triangles)
    params = np.full((len(triangles), 1), float(pressure))
    return ElementSet(name, PressureKernel(), triangles, params)
