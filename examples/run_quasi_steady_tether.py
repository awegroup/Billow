# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# SPDX-License-Identifier: Apache-2.0

"""Quasi-steady tether: the Tethers.jl examples, solved as an energy minimum.

Two cases, mirroring ``examples/quasisteady/`` of
`Tethers.jl <https://github.com/ufechner7/Tethers.jl>`_, with its default
tether (4 mm Dyneema, EA = 614.6 kN, cd = 0.958) and its load model:

1. ``catenary`` -- a kite fixed at [100, 100, 800] m on a tether 5% longer
   than the distance, 22 segments (``run_catenary.jl``). Gravity only, so the
   whole load is a potential and the shape is ONE solve. Compared with the
   analytic catenary through the same two points.
2. ``circular`` -- the kite flies one revolution of a 10 degree cone at
   0.05 rad/s, 500 m out, 20 segments, sampled every 0.02 s
   (``flying_circular.jl``). At each sample the kite node is moved and the
   tether re-solved from the previous shape.

Where each quasi-steady load lives
----------------------------------
Billow minimises a potential, so a load can be exact inside the solve only if
it has one. The three loads of the Tethers.jl model split as follows:

* **weight** -- a dead nodal force. Exact.
* **centrifugal term** ``-m omega x (omega x r)`` of a tether rotating rigidly
  with the kite position vector -- the gradient of ``U = -1/2 m |omega x r|^2``,
  so it is written here as a one-node element kernel, :class:`CentrifugalKernel`,
  with ``omega`` a live parameter. Exact, no iteration, no rebuild.
* **segment drag** ``c |v_n| v_n`` on the crossflow component of the apparent
  wind -- velocity-dependent and NOT conservative (its Jacobian is not
  symmetric), so no potential has it as gradient. It is applied as a dead load
  evaluated on the previous step's shape: one solve per step, no inner loop.

What that one-step lag costs was measured against a damped Newton solve of the
force balance with the drag inside the residual (built from
``PotentialEnergy.internal_load`` plus the same drag law in CasADi), over the
``circular`` sweep: the kite force differs by at most 3e-3 N on 57 N and the
nodes by 1.1 mm, with the two agreeing to 5e-10 N for a kite at rest. The
Newton solve itself is not the better tool here -- it needs ~10 iterations per
step against IPOPT's 4, and from a cold start its residual-norm line search
collapses to a hundredth of a step, because a 2 m transverse move of a taut
26 m segment adds 8 cm of stretch. The energy is a merit function; a residual
norm is not.

Two things a port of this model has to get right
------------------------------------------------
* The cables are **two-way** springs, ``tension_only=False``. Tethers.jl
  has neither branch: its ``l = (|F|/EA + 1) Ls`` lays every segment along
  the accumulated force, so a segment can only ever be taut, and on that
  branch the two laws coincide. A hanging tether never leaves it (the
  solved tensions are all positive), and the slack cut would cost the cold
  start: a catenary's chords are shorter than its arcs, so on the seed every
  curved segment is slack, i.e. of zero stiffness, and IPOPT stops with
  ``Error_In_Step_Computation``.
* The seed is the analytic catenary, spaced by **arc length**. Equal
  horizontal spacing on a near-vertical tether puts most of the length into a
  few segments and starts them stretched several-fold.

The drag law is local to this example on purpose: Billow imports NumPy and
CasADi only, and aerodynamics belongs on the other side of that boundary.

Run from the project root::

    python examples/run_quasi_steady_tether.py
    python examples/run_quasi_steady_tether.py --case circular --steps 600 --show
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from billow import MinimumEnergySolver, StructuralModel, StructuralSolution
from billow.elements import build_cable_elements, line_tensions
from billow.elements.base import ElementSet, element_translations
from billow.plotting import PALETTE, set_plot_style

OUTPUT_DIR = Path("results/demo")

# Tethers.jl ``StaticSettings`` defaults
RHO_AIR = 1.225  # kg/m^3
GRAVITY = 9.81  # m/s^2
CD_TETHER = 0.958
D_TETHER = 4.0e-3  # m
RHO_TETHER = 724.0  # kg/m^3, Dyneema
C_SPRING = 614600.0  # N, E A
TETHER_SLACK = 0.05  # unstretched length = (1 + slack) * kite distance
AREA = np.pi / 4 * D_TETHER**2


# ---------------------------------------------------------------------------
# Loads
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CentrifugalKernel:
    """One node in a frame rotating at ``omega`` about the origin.

    ``U = -1/2 m |omega x r|^2``, so ``-dU/dr = m (|omega|^2 r - (omega . r) omega)
    = -m omega x (omega x r)``: the centrifugal force, as a potential.
    """

    nodes_per_element: int = 1
    rotational_nodes: tuple[int, ...] = ()
    param_names: tuple[str, ...] = ("mass", "omega_x", "omega_y", "omega_z")

    def energy(self, q, frames, p):
        r = element_translations(q, 0)
        return -0.5 * p[0] * ca.sumsqr(ca.cross(p[1:4], r))


def kite_kinematics(kite_pos, kite_vel):
    """``(p_unit, v_parallel, omega)``: Tethers.jl ``node_kinematics`` inputs.

    The tether is taken to rotate rigidly with the kite position vector, at the
    angular velocity ``omega = p x v / |p|^2``, plus the radial kite speed.
    """
    distance = np.linalg.norm(kite_pos)
    p_unit = kite_pos / distance
    return p_unit, float(np.dot(kite_vel, p_unit)), np.cross(kite_pos / distance**2, kite_vel)


def crossflow_drag(v_app, direction, drag_coeff):
    """Tethers.jl ``segment_drag``: ``c |v_n| v_n`` on the crossflow component.

    Row-wise over ``(n, 3)`` arrays. ``drag_coeff = -1/2 rho Ls d cd`` carries
    the sign. Zero below 1 mm/s of apparent wind and when the wind is aligned
    with the segment, as in the reference.
    """
    v_n = v_app - np.sum(v_app * direction, axis=-1, keepdims=True) * direction
    n2 = np.sum(v_n**2, axis=-1, keepdims=True)
    still = np.all(np.abs(v_app) < 1e-3, axis=-1, keepdims=True) | (n2 < 1e-24)
    return np.where(still, 0.0, drag_coeff * np.sqrt(n2) * v_n)


def node_velocity(positions, kinematics):
    """Tethers.jl ``node_kinematics``: rigid rotation with the kite, plus reel."""
    p_unit, v_parallel, omega = kinematics
    return v_parallel * p_unit + np.cross(omega, positions)


def segment_drag(positions, kinematics, wind_vel, drag_coeff):
    """Drag of the segment above each interior node, lumped at that node."""
    pos = positions[1:-1]
    segment = positions[2:] - pos
    direction = segment / np.linalg.norm(segment, axis=1, keepdims=True)
    v_app = node_velocity(pos, kinematics) - wind_vel[1:-1]
    forces = np.zeros_like(positions)
    forces[1:-1] = crossflow_drag(v_app, direction, drag_coeff)
    return forces


# ---------------------------------------------------------------------------
# The analytic catenary, as seed and as reference
# ---------------------------------------------------------------------------


def analytic_catenary(kite_pos, length):
    """Inextensible catenary from the origin to ``kite_pos`` of arc length ``length``.

    Returns ``(a, x_min)``: the parameter ``a = H / w`` and the horizontal
    abscissa of the lowest point, measured along the kite's azimuth. The curve
    is ``z(r) = a cosh((r - x_min)/a) - a cosh(-x_min/a)`` and the tension
    ``T(r) = w a cosh((r - x_min)/a)``.
    """
    h = np.linalg.norm(kite_pos[:2])
    v = kite_pos[2]
    unfolded = np.sqrt(length**2 - v**2)
    a = brentq(lambda a: 2 * a * np.sinh(h / (2 * a)) - unfolded, h / 1400 + 1e-9, 1e7)
    x_min = -0.5 * (a * np.log((length + v) / (length - v)) - h)
    return a, x_min


def catenary_curve(kite_pos, length, n_points):
    a, x_min = analytic_catenary(kite_pos, length)
    r = np.linspace(0.0, np.linalg.norm(kite_pos[:2]), n_points)
    return r, a * np.cosh((r - x_min) / a) - a * np.cosh(-x_min / a)


def catenary_seed(kite_pos, length, n_nodes):
    """Nodes on the analytic catenary, equally spaced in ARC length."""
    a, x_min = analytic_catenary(kite_pos, length)
    s0 = a * np.sinh(-x_min / a)
    s = np.linspace(0.0, length, n_nodes)
    r = x_min + a * np.arcsinh((s + s0) / a)
    z = a * np.cosh((r - x_min) / a) - a * np.cosh(-x_min / a)
    azimuth = kite_pos[:2] / np.linalg.norm(kite_pos[:2])
    return np.column_stack([r * azimuth[0], r * azimuth[1], z])


# ---------------------------------------------------------------------------
# The tether
# ---------------------------------------------------------------------------


class QuasiSteadyTether:
    """A chain of ``segments`` two-way cables, ground node 0 pinned at the
    origin and the kite node pinned wherever the trajectory puts it.

    Mass lumping follows Tethers.jl: one segment mass per interior node, one
    and a half on the node nearest the ground.
    """

    def __init__(self, segments: int, kite_pos: np.ndarray) -> None:
        self.segments = segments
        n_nodes = segments + 1
        self.kite = n_nodes - 1
        self.interior = np.arange(1, self.kite)
        self.lumping = np.where(self.interior == 1, 1.5, 1.0)

        length = (1 + TETHER_SLACK) * np.linalg.norm(kite_pos)
        nodes = catenary_seed(kite_pos, length, n_nodes)
        self.segment_length = length / segments
        connectivity = np.column_stack([np.arange(segments), np.arange(1, n_nodes)])
        cables = build_cable_elements(
            connectivity,
            np.full(segments, self.segment_length),
            np.full(segments, C_SPRING / self.segment_length),
            tension_only=False,
        )
        centrifugal = ElementSet(
            "centrifugal",
            CentrifugalKernel(),
            self.interior[:, None],
            np.column_stack([self.mass, np.zeros((len(self.interior), 3))]),
        )
        self.model = StructuralModel(
            nodes, [cables, centrifugal], fixed_translation_nodes=[0, self.kite]
        )
        self.solver = MinimumEnergySolver(self.model, force_tolerance=1e-3)
        self.state = self.model.initial_state()

    # -- parameters ---------------------------------------------------------

    @property
    def mass(self) -> np.ndarray:
        return self.lumping * RHO_TETHER * AREA * self.segment_length

    @property
    def drag_coeff(self) -> float:
        return -0.5 * RHO_AIR * self.segment_length * D_TETHER * CD_TETHER

    def set_length(self, tether_length: float) -> None:
        """Unstretched length: rest length, stiffness and mass per segment."""
        self.segment_length = tether_length / self.segments
        cables = self.model.element_set("cables")
        cables = cables.with_param_column(
            "rest_length", np.full(self.segments, self.segment_length)
        )
        cables = cables.with_param_column(
            "stiffness", np.full(self.segments, C_SPRING / self.segment_length)
        )
        centrifugal = self.model.element_set("centrifugal").with_param_column(
            "mass", self.mass
        )
        self.model = self.model.replaced(cables).replaced(centrifugal)

    def set_omega(self, omega: np.ndarray) -> None:
        centrifugal = self.model.element_set("centrifugal")
        for column, value in zip(("omega_x", "omega_y", "omega_z"), omega):
            centrifugal = centrifugal.with_param_column(
                column, np.full(centrifugal.n_elements, value)
            )
        self.model = self.model.replaced(centrifugal)

    def weight(self) -> np.ndarray:
        forces = np.zeros((self.kite + 1, 3))
        forces[self.interior, 2] = -self.mass * GRAVITY
        return forces

    # -- one quasi-steady step ------------------------------------------------

    def step(
        self,
        kite_pos: np.ndarray,
        kite_vel: np.ndarray,
        wind_vel: np.ndarray | None = None,
        *,
        tether_length: float | None = None,
        with_drag: bool = True,
    ) -> tuple[StructuralSolution, np.ndarray]:
        """Move the kite node and re-solve from the previous shape.

        Returns the solution and the force the kite exerts on the tether end,
        in Tethers.jl's ``force_kite`` convention: the tension of the last
        segment plus the loads it lumps at the kite end (one and a half
        segment masses of weight and centrifugal load, and the last segment's
        drag), so the two codes' curves are directly comparable.
        """
        kite_pos = np.asarray(kite_pos, dtype=float)
        kite_vel = np.asarray(kite_vel, dtype=float)
        if wind_vel is None:
            wind_vel = np.zeros((self.kite + 1, 3))
        if tether_length is None:
            tether_length = (1 + TETHER_SLACK) * np.linalg.norm(kite_pos)
        self.set_length(tether_length)
        kinematics = kite_kinematics(kite_pos, kite_vel)
        self.set_omega(kinematics[2])
        self.state.positions[self.kite] = kite_pos

        forces = self.weight()
        if with_drag:
            forces += segment_drag(self.state.positions, kinematics, wind_vel, self.drag_coeff)
        solution = self.solver.solve(forces, state=self.state, model=self.model)
        self.state = solution.state

        # internal_forces at the pinned kite node is the elastic force ON the
        # kite, pulling it down the tether; the force the kite exerts on the
        # tether end is its negative, pointing up the last segment.
        tension_on_end = -solution.internal_forces[self.kite]
        p_unit, v_parallel, omega = kinematics
        end_mass = 1.5 * RHO_TETHER * AREA * self.segment_length
        end_load = end_mass * (np.cross(omega, np.cross(omega, kite_pos)) + [0.0, 0.0, GRAVITY])
        end_drag = np.zeros(3)
        if with_drag:
            last = self.state.positions[self.kite] - self.state.positions[self.kite - 1]
            end_drag = crossflow_drag(
                node_velocity(kite_pos, kinematics) - wind_vel[self.kite],
                last / np.linalg.norm(last), self.drag_coeff,
            )
        return solution, tension_on_end + end_load - end_drag

    def tensions(self) -> np.ndarray:
        return line_tensions(self.state.positions, self.model.element_set("cables"))


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


def catenary_case(**_) -> plt.Figure:
    kite_pos = np.array([100.0, 100.0, 800.0])
    segments = 22
    length = (1 + TETHER_SLACK) * np.linalg.norm(kite_pos)

    tether = QuasiSteadyTether(segments, kite_pos)
    started = time.perf_counter()
    solution, _ = tether.step(kite_pos, np.zeros(3))
    elapsed = time.perf_counter() - started
    print(f"  one solve: {elapsed * 1e3:.1f} ms, {solution.iterations} IPOPT iterations, "
          f"converged {solution.converged}, residual {solution.residual_norm:.1e} N")

    positions = tether.state.positions
    r_nodes = np.linalg.norm(positions[:, :2], axis=1)
    r_curve, z_curve = catenary_curve(kite_pos, length, 4000)
    distance = np.array(
        [np.hypot(r_curve - r, z_curve - z).min() for r, z in zip(r_nodes, positions[:, 2])]
    )
    a, x_min = analytic_catenary(kite_pos, length)
    weight_per_length = RHO_TETHER * AREA * GRAVITY
    tensions = tether.tensions()
    reference = weight_per_length * a * np.cosh((np.array([0.0, r_nodes[-1]]) - x_min) / a)
    print(f"  tension at the ground {tensions[0]:.2f} N (analytic {reference[0]:.2f}), "
          f"at the kite {tensions[-1]:.2f} N (analytic {reference[1]:.2f})")
    print(f"  largest node distance to the analytic catenary {distance.max():.2f} m at node "
          f"{distance.argmax()}: the chain is coarse where the tether is nearly slack, "
          f"and Tethers.jl's 1.5 mass lumping at node 1 pulls that node down")

    figure = plt.figure(figsize=(9.0, 5.0))
    grid = figure.add_gridspec(1, 3, width_ratios=[1.0, 1.4, 1.2])
    left = figure.add_subplot(grid[0])
    left.plot(r_curve, z_curve, color=PALETTE["Black"], label="analytic catenary")
    left.plot(r_nodes, positions[:, 2], marker="o", markersize=3.5, linestyle="-",
              color=PALETTE["Vermillion"], label=f"Billow, {segments} segments")
    left.set_xlabel("horizontal distance [m]")
    left.set_ylabel("height [m]")
    left.set_aspect("equal", adjustable="datalim")
    left.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0))

    # the lowest 120 m, where the two differ: the tether is nearly slack there
    middle = figure.add_subplot(grid[1])
    middle.plot(r_curve, z_curve, color=PALETTE["Black"])
    middle.plot(r_nodes, positions[:, 2], marker="o", markersize=4, linestyle="-",
                color=PALETTE["Vermillion"])
    middle.set_xlim(-5.0, 90.0)
    middle.set_ylim(-15.0, 120.0)
    middle.set_xlabel("horizontal distance [m]")
    middle.set_title("near the ground station", fontsize=9)
    middle.set_aspect("equal", adjustable="box")

    right = figure.add_subplot(grid[2], projection="3d")
    right.plot(*positions.T, marker="o", markersize=3, color=PALETTE["Vermillion"], label="tether")
    right.scatter(0, 0, 0, marker="s", s=40, color=PALETTE["Black"], label="ground station")
    right.scatter(*kite_pos, marker="D", s=30, color=PALETTE["Bluish Green"], label="kite")
    right.set_xlabel("x [m]")
    right.set_ylabel("y [m]")
    right.set_zlabel("z [m]")
    right.set_box_aspect((1.0, 1.0, 2.4))
    right.locator_params(nbins=4)
    right.legend(loc="upper left")
    figure.suptitle("Quasi-steady tether at rest against the analytic catenary")
    figure.tight_layout()
    return figure


def circular_case(steps: int | None = None, **_) -> plt.Figure:
    average_elevation = np.deg2rad(70.0)
    cone_angle = np.deg2rad(10.0)
    distance = 500.0
    gamma_dot = 0.05  # rad/s
    dt = 0.02  # s, as in Tethers.jl examples/Tether_11.jl
    segments = 20

    duration = 2 * np.pi / gamma_dot
    n_samples = len(np.arange(0.0, duration + 1e-12, dt))
    gamma = np.linspace(0.0, 2 * np.pi, n_samples)
    if steps is not None:
        gamma = gamma[:steps]
    rotation = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(average_elevation), -np.sin(average_elevation)],
        [0.0, np.sin(average_elevation), np.cos(average_elevation)],
    ])
    radius = distance * np.sin(cone_angle)
    trajectory = np.column_stack([
        radius * np.cos(gamma), radius * np.sin(gamma),
        np.full_like(gamma, distance * np.cos(cone_angle)),
    ]) @ rotation.T
    velocity = np.column_stack([
        -radius * np.sin(gamma) * gamma_dot, radius * np.cos(gamma) * gamma_dot,
        np.zeros_like(gamma),
    ]) @ rotation.T

    tether = QuasiSteadyTether(segments, trajectory[0])
    started = time.perf_counter()
    tether.step(trajectory[0], velocity[0], with_drag=False)  # catenary + centrifugal
    solution, _ = tether.step(trajectory[0], velocity[0])      # drag on that shape
    print(f"  cold start: {(time.perf_counter() - started) * 1e3:.1f} ms in two solves, "
          f"converged {solution.converged}")

    n = len(gamma)
    shapes = np.zeros((n, segments + 1, 3))
    force_kite = np.zeros((n, 3))
    force_ground = np.zeros(n)
    iterations = np.zeros(n, dtype=int)
    started = time.perf_counter()
    for i in range(n):
        solution, force_kite[i] = tether.step(trajectory[i], velocity[i])
        shapes[i] = tether.state.positions
        force_ground[i] = tether.tensions()[0]
        iterations[i] = solution.iterations
        if not solution.converged:
            print(f"  step {i}: {solution.status}, residual {solution.residual_norm:.2e} N")
    elapsed = time.perf_counter() - started
    print(f"  Elapsed time: {elapsed:.3f} s, speed: {n * dt / elapsed:.1f} times real-time, "
          f"simulations: {n}  ({elapsed / n * 1e3:.2f} ms per step, IPOPT iterations "
          f"mean {iterations.mean():.1f}, max {iterations.max()})")

    figure = plt.figure(figsize=(10.0, 4.4))
    left = figure.add_subplot(1, 2, 1, projection="3d")
    left.plot(*trajectory.T, color=PALETTE["Blue"], label="kite trajectory")
    stride = max(1, n // 20)
    for i in range(0, n, stride):
        left.plot(*shapes[i].T, color=PALETTE["Orange"], linewidth=0.8, linestyle=":",
                  marker="x", markersize=2.5,
                  label="tether" if i == 0 else None)
    left.scatter(0, 0, 0, marker="s", s=40, color=PALETTE["Black"], label="ground station")
    left.set_xlabel("x [m]")
    left.set_ylabel("y [m]")
    left.set_zlabel("z [m]")
    extent = np.concatenate([shapes.reshape(-1, 3), np.zeros((1, 3))])
    left.set_box_aspect(np.ptp(extent, axis=0) + 1.0)
    left.locator_params(nbins=4)
    left.legend(loc="upper left")

    right = figure.add_subplot(1, 2, 2)
    for k, label in enumerate(("$F_x$", "$F_y$", "$F_z$")):
        right.plot(gamma, force_kite[:, k] / 1e3, label=label)
    right.plot(gamma, force_ground / 1e3, color=PALETTE["Black"], linestyle="--",
               label="tension at the ground")
    right.set_xlabel(r"$\gamma$ [rad]")
    right.set_ylabel("force [kN]")
    right.legend()
    figure.suptitle("Tether force at the kite along a circular trajectory")
    figure.tight_layout()
    return figure


CASES = {"catenary": catenary_case, "circular": circular_case}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", choices=sorted(CASES), action="append",
                        help="run only these cases (default: all)")
    parser.add_argument("--steps", type=int, default=None,
                        help="circular: number of trajectory samples (default: one revolution)")
    parser.add_argument("--show", action="store_true", help="open the figures")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no-title", action="store_true",
                        help="drop figure titles (for a paper, where the caption carries them)")
    arguments = parser.parse_args()
    if arguments.no_title:
        import matplotlib.figure
        matplotlib.figure.Figure.suptitle = lambda self, *a, **k: None

    set_plot_style()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    for name in arguments.case or sorted(CASES):
        print(f"[{name}]")
        started = time.perf_counter()
        figure = CASES[name](steps=arguments.steps)
        target = arguments.output_dir / f"quasi_steady_{name}.png"
        figure.savefig(target, dpi=160)
        print(f"  -> {target}  ({time.perf_counter() - started:.1f} s)")
        if not arguments.show:
            plt.close(figure)
    if arguments.show:
        plt.show()


if __name__ == "__main__":
    main()
