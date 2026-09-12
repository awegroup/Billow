# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# SPDX-License-Identifier: Apache-2.0

"""The wireframe fidelity: a line system of cables and pulleys.

These assert against closed-form statics -- a symmetric two-cable node, a
catenary-free straight pull, a frictionless sheave equalising its two arms --
and against the invariants the class exists to provide: state carried across
solves, rest lengths and stiffnesses addressable by the caller's own line index,
and one rope reported identically from either of its arms.
"""

import numpy as np
import pytest

from billow import build_line_system
from billow.wireframe import CABLES, PULLEYS, STRUTS, pulley_triplets

STIFFNESS = 1.0e5


def _two_cable_node():
    """Free node at the origin, held by two cables from x = +/-1."""
    nodes = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    return build_line_system(
        nodes,
        connectivity=[[0, 2], [1, 2]],
        rest_lengths=[1.0, 1.0],
        stiffness=[STIFFNESS, STIFFNESS],
        fixed_nodes=[0, 1],
    )


# -- statics ---------------------------------------------------------------


def test_straight_pull_matches_hookes_law():
    """One cable, pulled along its own axis: the answer is f = k (l - l0)."""
    system = build_line_system(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        connectivity=[[0, 1]],
        rest_lengths=[1.0],
        stiffness=[STIFFNESS],
        fixed_nodes=[0],
    )
    forces = np.array([[0.0, 0.0, 0.0], [250.0, 0.0, 0.0]])

    solution = system.solve(forces)

    assert solution.converged
    assert system.positions[1, 0] == pytest.approx(1.0 + 250.0 / STIFFNESS, rel=1e-8)
    assert system.tensions()[0] == pytest.approx(250.0, rel=1e-6)


def test_symmetric_pair_shares_the_load_equally():
    """A vertical load on a symmetric pair: both cables carry the same tension."""
    system = _two_cable_node()
    forces = np.zeros((3, 3))
    forces[2] = [0.0, 0.0, -400.0]

    solution = system.solve(forces)

    assert solution.converged
    tensions = system.tensions()
    assert tensions[0] == pytest.approx(tensions[1], rel=1e-8)
    # Vertical equilibrium of the free node: 2 T sin(theta) = 400.
    sag = -system.positions[2, 2]
    sin_theta = sag / np.linalg.norm(system.positions[2] - system.positions[0])
    assert 2.0 * tensions[0] * sin_theta == pytest.approx(400.0, rel=1e-6)


def test_a_slack_line_carries_nothing():
    """Push the free node toward one anchor: that cable goes slack, not compressive."""
    system = _two_cable_node()
    forces = np.zeros((3, 3))
    forces[2] = [-300.0, 0.0, 0.0]

    system.solve(forces)

    tensions = system.tensions()
    assert tensions[0] == pytest.approx(0.0, abs=1e-9)  # pushed-toward anchor
    assert tensions[1] == pytest.approx(300.0, rel=1e-6)


def test_a_two_way_line_can_take_compression():
    """``tension_only=False`` keeps the compressive branch, and says so."""
    system = build_line_system(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        connectivity=[[0, 1]],
        rest_lengths=[1.0],
        stiffness=[STIFFNESS],
        tension_only=False,
        fixed_nodes=[0],
    )
    forces = np.array([[0.0, 0.0, 0.0], [-250.0, 0.0, 0.0]])

    system.solve(forces)

    assert system.line_set == [STRUTS]
    assert system.positions[1, 0] == pytest.approx(1.0 - 250.0 / STIFFNESS, rel=1e-8)
    assert system.tensions()[0] == pytest.approx(-250.0, rel=1e-6)


# -- pulleys ---------------------------------------------------------------


def _pulley_system():
    """A rope over a sheave, anchored at both ends, loaded at the sheave."""
    nodes = np.array([[-1.0, 0.0, 0.0], [0.1, -0.5, 0.0], [1.0, 0.0, 0.0]])
    return build_line_system(
        nodes,
        connectivity=[[0, 1], [1, 2]],
        rest_lengths=[1.2, 1.2],
        stiffness=[STIFFNESS, STIFFNESS],
        pulley_arm_pairs=[(0, 1)],
        fixed_nodes=[0, 2],
    )


def test_pulley_triplets_put_the_sheave_in_the_middle():
    triplets = pulley_triplets(np.array([[0, 1], [1, 2]]), [(0, 1)])
    assert triplets == [(0, 1, 2)]


def test_pulley_triplets_reject_arms_that_do_not_meet():
    with pytest.raises(ValueError, match="share 0 nodes"):
        pulley_triplets(np.array([[0, 1], [2, 3]]), [(0, 1)])


def test_a_frictionless_sheave_equalises_its_two_arms():
    system = _pulley_system()
    forces = np.zeros((3, 3))
    forces[1] = [0.0, -300.0, 0.0]

    solution = system.solve(forces)

    assert solution.converged
    # Both arm indices address the one rope, so both report its one tension.
    tensions = system.tensions()
    assert tensions[0] == pytest.approx(tensions[1], rel=1e-12)
    # And that tension is what each anchor feels.
    assert tensions[0] == pytest.approx(
        np.linalg.norm(solution.internal_forces[0]), rel=1e-6
    )
    assert tensions[0] == pytest.approx(
        np.linalg.norm(solution.internal_forces[2]), rel=1e-6
    )


def test_a_pulley_rest_length_is_the_whole_rope():
    """The arms are summed by default, and both indices report the total."""
    system = _pulley_system()

    assert system.rest_length(0) == pytest.approx(2.4)
    assert system.rest_length(1) == pytest.approx(2.4)
    assert system.line_set == [PULLEYS, PULLEYS]


def test_explicit_pulley_rest_length_overrides_the_sum():
    """A format storing the rope total on each arm must say so, not be inferred."""
    nodes = np.array([[-1.0, 0.0, 0.0], [0.1, -0.5, 0.0], [1.0, 0.0, 0.0]])
    system = build_line_system(
        nodes,
        connectivity=[[0, 1], [1, 2]],
        rest_lengths=[2.4, 2.4],  # the total, stored on each arm
        stiffness=[STIFFNESS, STIFFNESS],
        pulley_arm_pairs=[(0, 1)],
        pulley_rest_lengths=[2.4],
        fixed_nodes=[0, 2],
    )

    assert system.rest_length(0) == pytest.approx(2.4)


def test_the_pulley_length_is_both_arms_summed():
    system = _pulley_system()
    lengths = system.lengths()

    expected = np.linalg.norm(
        system.positions[1] - system.positions[0]
    ) + np.linalg.norm(system.positions[2] - system.positions[1])
    assert lengths[0] == pytest.approx(expected)
    assert lengths[1] == pytest.approx(expected)


# -- state ownership -------------------------------------------------------


def test_state_carries_across_solves():
    """The second solve starts from the first one's answer, not from the build."""
    system = _two_cable_node()
    forces = np.zeros((3, 3))
    forces[2] = [0.0, 0.0, -400.0]

    system.solve(forces)
    after_first = system.positions.copy()
    second = system.solve(forces)

    np.testing.assert_allclose(second.state.positions, after_first, atol=1e-9)
    # Same load, already at equilibrium: nothing left to do.
    assert second.residual_norm < 1e-6


def test_shortening_a_line_pulls_the_node_toward_its_anchor():
    """Actuation is a rest-length change between solves, with no rebuild."""
    system = _two_cable_node()
    forces = np.zeros((3, 3))
    forces[2] = [0.0, 0.0, -400.0]
    system.solve(forces)
    solver = system.solver

    system.set_rest_length(0, 0.8)
    system.solve(forces)

    assert system.rest_length(0) == pytest.approx(0.8)
    assert system.positions[2, 0] < 0.0  # drawn toward the x = -1 anchor
    assert system.solver is solver  # the compiled graph was reused


def test_stiffnesses_round_trip_through_the_caller_ordering():
    system = _pulley_system()

    values = system.stiffnesses
    np.testing.assert_allclose(values, [STIFFNESS, STIFFNESS])

    system.stiffnesses = values * 0.5
    np.testing.assert_allclose(system.stiffnesses, [0.5 * STIFFNESS] * 2)


def test_a_stiffness_ramp_reuses_one_compiled_solver():
    system = _two_cable_node()
    forces = np.zeros((3, 3))
    forces[2] = [0.0, 0.0, -400.0]
    system.solve(forces)
    solver = system.solver

    for factor in (0.1, 0.5, 1.0):
        system.stiffnesses = np.full(system.n_lines, STIFFNESS * factor)
        system.solve(forces)
        assert system.solver is solver

    # Softer lines sag more; the last ramp step is back at full stiffness.
    assert system.rest_lengths[0] == pytest.approx(1.0)


# -- bookkeeping -----------------------------------------------------------


def test_dropped_lines_keep_their_place_in_the_numbering():
    """A line another element type carries is dropped, not renumbered away."""
    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    system = build_line_system(
        nodes,
        connectivity=[[0, 1], [1, 2]],
        rest_lengths=[1.0, 1.0],
        stiffness=[STIFFNESS, STIFFNESS],
        drop_lines=[0],
        fixed_nodes=[0],
    )

    assert system.n_lines == 2
    assert system.line_set == ["", CABLES]
    assert np.isnan(system.stiffnesses[0])
    with pytest.raises(KeyError, match="dropped"):
        system.rest_length(0)


def test_line_counts_must_agree():
    with pytest.raises(ValueError, match="line count"):
        build_line_system(
            np.zeros((2, 3)),
            connectivity=[[0, 1]],
            rest_lengths=[1.0, 1.0],
            stiffness=[STIFFNESS],
        )


def test_a_line_cannot_be_an_arm_of_two_pulleys():
    nodes = np.zeros((4, 3))
    with pytest.raises(ValueError, match="already an arm"):
        build_line_system(
            nodes,
            connectivity=[[0, 1], [1, 2], [1, 3]],
            rest_lengths=[1.0] * 3,
            stiffness=[STIFFNESS] * 3,
            pulley_arm_pairs=[(0, 1), (1, 2)],
        )


def test_solver_settings_cannot_be_changed_after_the_build():
    system = _two_cable_node()
    system.solve(np.zeros((3, 3)))

    with pytest.raises(ValueError, match="fixed at build time"):
        system.solve(np.zeros((3, 3)), tolerance=1e-4)
