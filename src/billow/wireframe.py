# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
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

"""The wireframe model: a line system of cables and frictionless pulleys.

This is Billow's simplified fidelity -- a bridle, a line system, a tensile net:
everything the structure carries in axial tension, and nothing it carries in
bending. It is not a separate code path. It builds the *same*
:class:`~billow.model.StructuralModel` out of the *same*
:class:`~billow.elements.cable.CableKernel` and
:class:`~billow.elements.cable.PulleyKernel` the full model uses, so the two
fidelities cannot drift apart: there is one cable force law in this package, and
both fidelities call it.

What this class adds on top of the raw element sets is **state ownership over a
sequence of solves**, which is what an actuated or iteratively coupled run
needs:

* positions persist from one solve to the next, so a fixed-point loop warm
  starts naturally;
* rest lengths are addressable by the caller's own line index and can be
  changed between solves (this is how tape actuation works);
* stiffnesses likewise, which is how a stiffness ramp works.

None of those change the topology, so one compiled solver serves the whole run
-- see "Everything numeric is a parameter" in ``AGENTS.md``.

Lines and elements are not one-to-one
-------------------------------------
A pulley rope is ONE three-node element here, but callers typically carry it as
TWO two-node rows (the arms either side of the sheave), because that is how line
systems are usually tabulated. Tension-only and two-way lines also compile to
different kernels, so they land in different element sets.

:class:`LineSystem` therefore keeps a map from the caller's line index onto
``(element set, row)``, and every accessor speaks the caller's index. For a
pulley, both arm indices address the one rope, and its rest length is the length
of the WHOLE rope, both arms together.

That last point is the convention to state explicitly whenever a file format is
adapted onto this class, because it is exactly where two otherwise-agreeing
codes disagree: some formats store the rope total on each arm, others split it
across the two.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .elements import build_cable_elements, build_pulley_elements, line_tensions
from .elements.base import ElementSet
from .model import StructuralModel, StructuralState
from .solver import MinimumEnergySolver, StructuralSolution

Array = np.ndarray

logger = logging.getLogger(__name__)

#: Element-set names, so a caller can reach a set by name on the model.
CABLES = "cables"  # tension-only lines: rope, webbing, bridle
STRUTS = "struts"  # two-way axial lines, kept apart because it is another kernel
PULLEYS = "pulleys"


@dataclass
class LineSystem:
    """A cable and pulley network, and the state it owns across solves.

    Build it with :func:`build_line_system` rather than by hand -- the
    constructor takes an already-assembled model and the index map relating it
    to the caller's line numbering.

    Attributes
    ----------
    model
        The current :class:`~billow.model.StructuralModel`. It is REPLACED, not
        mutated, whenever a rest length or a stiffness changes, because element
        parameters live in frozen dataclasses. Always read it off the instance.
    state
        The live configuration. After :meth:`solve` this is the solved one, so
        consecutive solves continue from where the last left off.
    line_set
        Element-set name carrying each caller line, or ``""`` for a line that
        was dropped.
    line_row
        Row within that set, or ``-1`` for a dropped line.
    """

    model: StructuralModel
    state: StructuralState
    line_set: list[str]
    line_row: Array
    solver: MinimumEnergySolver | None = field(default=None, repr=False)
    last_solution: StructuralSolution | None = field(default=None, repr=False)

    # -- geometry ---------------------------------------------------------

    @property
    def positions(self) -> Array:
        """Current node positions ``(n_nodes, 3)`` [m]."""
        return self.state.positions

    @property
    def n_lines(self) -> int:
        """Number of lines in the caller's numbering."""
        return len(self.line_set)

    @property
    def is_pulley(self) -> Array:
        """Boolean mask over caller line indices: is this line a pulley arm?"""
        return np.array([name == PULLEYS for name in self.line_set])

    def _locate(self, line: int) -> tuple[str, int]:
        """Resolve a caller line index onto ``(set name, row)``."""
        name = self.line_set[line]
        if not name:
            raise KeyError(
                f"line {line} is not represented in this model (it was dropped "
                f"when the system was built)"
            )
        return name, int(self.line_row[line])

    def _rows_by_set(self) -> dict[str, tuple[Array, Array]]:
        """``set name -> (caller line indices, rows)`` for every built line."""
        grouped: dict[str, tuple[list[int], list[int]]] = {}
        for line, name in enumerate(self.line_set):
            if not name:
                continue
            lines, rows = grouped.setdefault(name, ([], []))
            lines.append(line)
            rows.append(int(self.line_row[line]))
        return {
            name: (np.array(lines, dtype=int), np.array(rows, dtype=int))
            for name, (lines, rows) in grouped.items()
        }

    # -- the parameters an actuated run changes ---------------------------

    def rest_length(self, line: int) -> float:
        """Rest length [m] of the line at the caller's index.

        For a pulley arm this is the length of the whole rope, both arms, which
        is what the kernel holds. Asking either arm gives the same number.
        """
        name, row = self._locate(line)
        return float(self.model.element_set(name).param_column("rest_length")[row])

    def set_rest_length(self, line: int, value: float) -> None:
        """Set one rest length [m]. A parameter update, never a rebuild.

        On a pulley arm this sets the whole rope, so setting both arms of one
        rope applies the second value, not their sum.
        """
        name, row = self._locate(line)
        element_set = self.model.element_set(name)
        lengths = element_set.param_column("rest_length").copy()
        lengths[row] = float(value)
        self.model = self.model.replaced(
            element_set.with_param_column("rest_length", lengths)
        )

    @property
    def rest_lengths(self) -> Array:
        """Rest length [m] per caller line index, ``NaN`` where there is none."""
        return self._gather("rest_length")

    @property
    def stiffnesses(self) -> Array:
        """Stiffness [N/m] per caller line index, ``NaN`` where there is none.

        A pulley rope's stiffness appears at BOTH of its arm indices, since both
        address the same element.
        """
        return self._gather("stiffness")

    def _gather(self, column: str) -> Array:
        values = np.full(self.n_lines, np.nan)
        for name, (lines, rows) in self._rows_by_set().items():
            values[lines] = self.model.element_set(name).param_column(column)[rows]
        return values

    @stiffnesses.setter
    def stiffnesses(self, values: Array) -> None:
        self.set_stiffnesses(values)

    def set_stiffnesses(self, values: Array) -> None:
        """Set every stiffness [N/m] at once, in the caller's line order.

        ``NaN`` entries are ignored, so the array :attr:`stiffnesses` returns
        can be scaled and handed straight back. A pulley's two arms address one
        element; they should carry the same value, and the later index wins if
        they do not.
        """
        values = np.asarray(values, dtype=float)
        if values.shape != (self.n_lines,):
            raise ValueError(
                f"expected one stiffness per line ({self.n_lines}), got {values.shape}"
            )
        for name, (lines, rows) in self._rows_by_set().items():
            element_set = self.model.element_set(name)
            column = element_set.param_column("stiffness").copy()
            known = np.isfinite(values[lines])
            column[rows[known]] = values[lines[known]]
            self.model = self.model.replaced(
                element_set.with_param_column("stiffness", column)
            )

    # -- solving ----------------------------------------------------------

    def build_solver(self, **settings: Any) -> MinimumEnergySolver:
        """Compile the solver for this topology and keep it.

        Settings go straight to :class:`~billow.solver.MinimumEnergySolver`.
        Calling this again recompiles, which is only needed if the topology
        changed -- and it cannot change through this class.
        """
        self.solver = MinimumEnergySolver(self.model, **settings)
        return self.solver

    def solve(
        self,
        external_forces: Array,
        *,
        warm_start: bool = True,
        **settings: Any,
    ) -> StructuralSolution:
        """Solve for static equilibrium under ``external_forces`` ``(n_nodes, 3)`` [N].

        The solved configuration is absorbed into :attr:`state`, so the next
        call continues from it. ``settings`` build the solver on the first call
        only; passing them once one exists is an error rather than a silently
        ignored request.
        """
        if self.solver is None:
            self.build_solver(**settings)
        elif settings:
            raise ValueError(
                "solver settings are fixed at build time; call build_solver() "
                "before the first solve, or omit them here"
            )

        solution = self.solver.solve(
            np.asarray(external_forces, dtype=float),
            state=self.state,
            model=self.model,
            warm_start=warm_start,
        )
        self.state = solution.state
        self.last_solution = solution
        return solution

    # -- diagnostics ------------------------------------------------------

    def tensions(self) -> Array:
        """Tension [N] per caller line index, ``NaN`` where there is no line.

        Read off the kernel's own energy gradient at the current positions, so
        the slack cut is exactly the solver's: a slack line reads 0, and a
        two-way line in compression reads negative. Both arms of a pulley report
        the single rope tension, which is the point of a frictionless sheave.

        Feed it SOLVED positions. On a stiff line a millimetre of spurious
        stretch is hundreds of newtons, so an interpolated shape gives a
        plausible-looking wrong answer.
        """
        values = np.full(self.n_lines, np.nan)
        for name, (lines, rows) in self._rows_by_set().items():
            element_set = self.model.element_set(name)
            if element_set.n_elements == 0:
                continue
            values[lines] = line_tensions(self.state.positions, element_set)[rows]
        return values

    def lengths(self) -> Array:
        """Current length [m] per caller line index, ``NaN`` where there is none.

        A pulley reports the whole rope, both arms summed, so it is directly
        comparable with :meth:`rest_length`.
        """
        values = np.full(self.n_lines, np.nan)
        positions = self.state.positions
        for name, (lines, rows) in self._rows_by_set().items():
            element_set = self.model.element_set(name)
            if element_set.n_elements == 0:
                continue
            corners = positions[element_set.connectivity]
            spans = np.linalg.norm(np.diff(corners, axis=1), axis=2).sum(axis=1)
            values[lines] = spans[rows]
        return values


def pulley_triplets(
    connectivity: Array, arm_pairs: Sequence[tuple[int, int]]
) -> list[tuple[int, int, int]]:
    """Turn pairs of arm line indices into ``(i, sheave, k)`` node triplets.

    Each pair names the two lines either side of one sheave. They must share
    exactly one node -- the sheave -- and that shared node becomes the middle of
    the triplet, which is the node
    :class:`~billow.elements.cable.PulleyKernel` equalises the tension about.

    Raises rather than guesses: a pair that does not share exactly one node is a
    pairing error in the caller's table, and picking an ordering anyway would
    build a plausible-looking wrong rope.
    """
    connectivity = np.asarray(connectivity, dtype=int)
    triplets: list[tuple[int, int, int]] = []
    for first, second in arm_pairs:
        a, b = (int(n) for n in connectivity[first])
        c, d = (int(n) for n in connectivity[second])
        shared = {a, b} & {c, d}
        if len(shared) != 1:
            raise ValueError(
                f"pulley arms {first} ({a}-{b}) and {second} ({c}-{d}) share "
                f"{len(shared)} nodes; exactly one (the sheave) is required"
            )
        sheave = shared.pop()
        triplets.append((b if a == sheave else a, sheave, d if c == sheave else c))
    return triplets


def build_line_system(
    nodes: Array,
    connectivity: Array,
    rest_lengths: Array,
    stiffness: Array,
    *,
    tension_only: Sequence[bool] | bool = True,
    pulley_arm_pairs: Sequence[tuple[int, int]] = (),
    pulley_rest_lengths: Sequence[float] | None = None,
    fixed_nodes: Sequence[int] = (),
    drop_lines: Sequence[int] = (),
    slack_smoothing: float = 0.0,
) -> LineSystem:
    """Assemble a :class:`LineSystem` from a table of lines.

    Parameters
    ----------
    nodes
        ``(n_nodes, 3)`` initial positions [m].
    connectivity
        ``(n_lines, 2)`` node indices. A pulley's two arms occupy two rows here,
        like every other line.
    rest_lengths, stiffness
        One entry per line, [m] and [N/m].
    tension_only
        Per line, or one value for all. ``True`` cuts compression, which is what
        a rope does; ``False`` leaves a two-way axial spring, which is what a
        strut or a test case sometimes wants.
    pulley_arm_pairs
        Pairs of line indices ``(i, j)`` that are the two arms of one rope. Each
        pair becomes a single three-node pulley element.
    pulley_rest_lengths
        The WHOLE-rope rest length [m] for each pair, in the same order. Omit it
        and the two arms' own rest lengths are summed -- right when the source
        table splits the rope across its arms, wrong when it stores the total on
        each arm. State which convention the source uses rather than letting it
        be inferred.
    fixed_nodes
        Node indices pinned in translation.
    drop_lines
        Line indices to leave out of the model while keeping their place in the
        caller's numbering (they report ``NaN`` and raise on access). Use it
        when another element type already carries that line -- a canopy spring
        replaced by membrane triangles, say.
    slack_smoothing
        Rounds the slack corner of the tension-only cut. ``0.0`` keeps the exact
        kink, which is C1 and is what the tests assert against.
    """
    nodes = np.asarray(nodes, dtype=float)
    connectivity = np.asarray(connectivity, dtype=int)
    rest_lengths = np.asarray(rest_lengths, dtype=float)
    stiffness = np.asarray(stiffness, dtype=float)
    n_lines = len(connectivity)

    if not (len(rest_lengths) == len(stiffness) == n_lines):
        raise ValueError(
            f"connectivity, rest_lengths and stiffness must agree on the line "
            f"count (got {n_lines}, {len(rest_lengths)}, {len(stiffness)})"
        )

    if isinstance(tension_only, bool):
        tension_only_arr = np.full(n_lines, tension_only)
    else:
        tension_only_arr = np.asarray(tension_only, dtype=bool)
        if len(tension_only_arr) != n_lines:
            raise ValueError(
                f"tension_only must be one flag or one per line ({n_lines}), "
                f"got {len(tension_only_arr)}"
            )

    arm_pairs = [(int(a), int(b)) for a, b in pulley_arm_pairs]
    is_pulley_arm = np.zeros(n_lines, dtype=bool)
    for first, second in arm_pairs:
        if is_pulley_arm[first] or is_pulley_arm[second]:
            raise ValueError(
                f"line {first} or {second} is already an arm of another pulley"
            )
        is_pulley_arm[[first, second]] = True

    is_dropped = np.zeros(n_lines, dtype=bool)
    is_dropped[list(drop_lines)] = True
    if (is_pulley_arm & is_dropped).any():
        raise ValueError("a line cannot be both a pulley arm and dropped")

    is_cable = ~(is_pulley_arm | is_dropped)
    one_way = np.flatnonzero(is_cable & tension_only_arr)
    two_way = np.flatnonzero(is_cable & ~tension_only_arr)

    line_set = [""] * n_lines
    line_row = np.full(n_lines, -1, dtype=int)
    element_sets: list[ElementSet] = []

    # An element set with no elements compiles to a degenerate `Function.map`,
    # so a set is only added once it has something in it. A system of pure
    # cables therefore carries no pulley set at all, and the accessors key off
    # `line_set`, which never names a set that was not built.
    def _add(name: str, indices: Array, builder) -> None:
        if not len(indices):
            return
        element_sets.append(builder())
        for row, line in enumerate(indices):
            line_set[int(line)] = name
            line_row[int(line)] = row

    # Tension-only and two-way lines are different kernels, so they are
    # different sets. Both are "cables" to the caller.
    _add(
        CABLES,
        one_way,
        lambda: build_cable_elements(
            connectivity[one_way],
            rest_lengths[one_way],
            stiffness[one_way],
            tension_only=True,
            slack_smoothing=slack_smoothing,
            name=CABLES,
        ),
    )
    _add(
        STRUTS,
        two_way,
        lambda: build_cable_elements(
            connectivity[two_way],
            rest_lengths[two_way],
            stiffness[two_way],
            tension_only=False,
            name=STRUTS,
        ),
    )

    triplets = pulley_triplets(connectivity, arm_pairs)
    if pulley_rest_lengths is None:
        rope_lengths = [
            float(rest_lengths[first] + rest_lengths[second])
            for first, second in arm_pairs
        ]
    else:
        rope_lengths = [float(value) for value in pulley_rest_lengths]
        if len(rope_lengths) != len(arm_pairs):
            raise ValueError(
                f"pulley_rest_lengths must have one entry per pair "
                f"({len(arm_pairs)}), got {len(rope_lengths)}"
            )

    if arm_pairs:
        element_sets.append(
            build_pulley_elements(
                np.asarray(triplets, dtype=int).reshape(-1, 3),
                rope_lengths,
                [float(stiffness[first]) for first, _ in arm_pairs],
                name=PULLEYS,
            )
        )
        for row, (first, second) in enumerate(arm_pairs):
            line_set[first] = line_set[second] = PULLEYS
            line_row[[first, second]] = row

    if not element_sets:
        raise ValueError("a line system needs at least one line that is not dropped")

    model = StructuralModel(
        nodes, element_sets, fixed_translation_nodes=sorted(int(n) for n in fixed_nodes)
    )

    logger.info(
        "Billow wireframe: %d nodes, %d DOF | %d cables, %d struts, %d pulleys, "
        "%d dropped",
        model.n_nodes,
        model.layout.n_dof,
        len(one_way),
        len(two_way),
        len(arm_pairs),
        int(is_dropped.sum()),
    )

    return LineSystem(
        model=model,
        state=model.initial_state(),
        line_set=line_set,
        line_row=line_row,
    )


__all__ = [
    "CABLES",
    "STRUTS",
    "PULLEYS",
    "LineSystem",
    "build_line_system",
    "pulley_triplets",
]
