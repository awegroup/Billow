# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# SPDX-License-Identifier: Apache-2.0

"""A hanging chain of cables against the analytic catenary.

The closed-form reference for a tether under its own weight. A chain of
two-way cables with the weight lumped at its nodes is a discretisation of the
inextensible catenary, so its nodes must converge onto the curve at O(h^2)
and its segment tensions onto the catenary tension at the segment midpoints.
This is what ``examples/run_quasi_steady_tether.py`` rests on.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import brentq

from billow import MinimumEnergySolver, StructuralModel
from billow.elements import build_cable_elements, line_tensions

WEIGHT_PER_LENGTH = 1.0  # N/m
AXIAL_STIFFNESS = 1.0e8  # N, E A: near inextensible at these tensions


def analytic_catenary(end: np.ndarray, length: float) -> tuple[float, float]:
    """``(a, x_min)`` of the inextensible catenary from the origin to ``end``.

    ``a = H / w`` and ``x_min`` the horizontal abscissa of the lowest point, so
    ``z(r) = a cosh((r - x_min)/a) - a cosh(-x_min/a)`` and
    ``T(r) = w a cosh((r - x_min)/a)``.
    """
    h = np.linalg.norm(end[:2])
    v = end[2]
    unfolded = np.sqrt(length**2 - v**2)
    a = brentq(lambda a: 2 * a * np.sinh(h / (2 * a)) - unfolded, h / 1400 + 1e-9, 1e7)
    x_min = -0.5 * (a * np.log((length + v) / (length - v)) - h)
    return a, x_min


def hanging_chain(end: np.ndarray, length: float, segments: int):
    """Solve a chain of ``segments`` cables from the origin to ``end``.

    Seeded on the analytic catenary, spaced by arc length; returns the solved
    positions, the segment tensions and the solution.
    """
    a, x_min = analytic_catenary(end, length)
    n_nodes = segments + 1
    s0 = a * np.sinh(-x_min / a)
    s = np.linspace(0.0, length, n_nodes)
    r = x_min + a * np.arcsinh((s + s0) / a)
    z = a * np.cosh((r - x_min) / a) - a * np.cosh(-x_min / a)
    azimuth = end[:2] / np.linalg.norm(end[:2])
    nodes = np.column_stack([r * azimuth[0], r * azimuth[1], z])

    segment_length = length / segments
    connectivity = np.column_stack([np.arange(segments), np.arange(1, n_nodes)])
    cables = build_cable_elements(
        connectivity,
        np.full(segments, segment_length),
        np.full(segments, AXIAL_STIFFNESS / segment_length),
        tension_only=False,
    )
    model = StructuralModel(nodes, [cables], fixed_translation_nodes=[0, n_nodes - 1])
    forces = np.zeros((n_nodes, 3))
    forces[1:-1, 2] = -WEIGHT_PER_LENGTH * segment_length
    solution = MinimumEnergySolver(model, force_tolerance=1e-6).solve(forces)
    positions = solution.state.positions
    return positions, line_tensions(positions, cables), solution


def chain_errors(end: np.ndarray, length: float, segments: int) -> tuple[float, float]:
    """Largest node height error [m] and relative segment-tension error."""
    a, x_min = analytic_catenary(end, length)
    positions, tensions, solution = hanging_chain(end, length, segments)
    assert solution.converged, solution.status

    r = np.linalg.norm(positions[:, :2], axis=1)
    z_exact = a * np.cosh((r - x_min) / a) - a * np.cosh(-x_min / a)
    height_error = float(np.abs(positions[:, 2] - z_exact).max())

    r_mid = 0.5 * (r[:-1] + r[1:])
    tension_exact = WEIGHT_PER_LENGTH * a * np.cosh((r_mid - x_min) / a)
    tension_error = float(np.abs(tensions - tension_exact).max() / tension_exact.max())
    return height_error, tension_error


@pytest.mark.parametrize(
    "end, slack",
    [
        (np.array([300.0, 0.0, 300.0]), 0.05),  # a kite tether
        (np.array([400.0, 0.0, 50.0]), 0.20),  # a deep, nearly horizontal span
    ],
)
def test_chain_converges_to_the_catenary_at_second_order(end, slack):
    length = (1 + slack) * np.linalg.norm(end)
    errors = [chain_errors(end, length, segments) for segments in (10, 20, 40)]
    heights = np.array([height for height, _ in errors])
    tensions = np.array([tension for _, tension in errors])

    # O(h^2) in the shape: doubling the segments quarters the error.
    ratios = heights[:-1] / heights[1:]
    assert np.all(ratios > 3.3), ratios
    # Tension at the segment midpoints is the catenary's, and improves with it.
    assert np.all(tensions[:-1] > tensions[1:]), tensions
    assert tensions[-1] < 5e-4, tensions
    assert heights[-1] < 1e-4 * length, heights


def test_chain_stays_in_the_vertical_plane():
    end = np.array([100.0, 100.0, 800.0])
    length = 1.05 * np.linalg.norm(end)
    positions, tensions, solution = hanging_chain(end, length, 40)
    assert solution.converged
    normal = np.array([-end[1], end[0], 0.0]) / np.linalg.norm(end[:2])
    assert np.abs(positions @ normal).max() < 1e-9
    # Every segment carries tension, and it grows monotonically towards the
    # upper end: the chain hangs, it does not fold.
    assert np.all(tensions > 0.0)
    assert np.all(np.diff(tensions) > 0.0)
