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

"""Billow -- minimum-energy structural modelling of soft, inflatable structures.

One formulation, two fidelities. Both pose static equilibrium as the
minimisation of a total potential energy and hand it to IPOPT, so a slack line
and a wrinkled panel are states of the minimiser rather than branches a Newton
residual has to be steered through.

    wireframe   cables, tension-only lines and frictionless pulleys -- the
                bridle and line system on its own
    full        the above plus geometrically exact Timoshenko beams,
                inflatable-tube constitutive laws and wrinkling CST membrane
                fabric -- the whole kite

The pipeline is the same either way::

    nodes + element sets  ->  StructuralModel
    StructuralModel       ->  MinimumEnergySolver   (compiled once)
    solver.solve(forces)  ->  StructuralSolution    (state, residual, energy)

Deliberately isolated: this package imports only NumPy and CasADi. It knows
nothing about aerodynamics, coupling loops or any particular kite file format.
Coupling it to a solver belongs in an adapter on the other side of that
boundary -- ``AWETrim``'s ``aerostructural/`` package is the reference one.

Example
-------
>>> import numpy as np
>>> from billow import MinimumEnergySolver, StructuralModel
>>> from billow.elements import build_cable_elements
>>> nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
>>> cables = build_cable_elements([[0, 1]], rest_lengths=[1.0], stiffness=[1e4])
>>> model = StructuralModel(nodes, [cables], fixed_translation_nodes=[0])
>>> solution = MinimumEnergySolver(model).solve(np.array([[0.0, 0, 0], [10.0, 0, 0]]))
>>> bool(solution.converged), round(float(solution.state.positions[1, 0]), 4)
(True, 1.001)
"""

__version__ = "0.1.0"

from .energy import PotentialEnergy
from .model import DofLayout, StructuralModel, StructuralState
from .solver import MinimumEnergySolver, StructuralSolution
from .wireframe import LineSystem, build_line_system

__all__ = [
    "DofLayout",
    "StructuralState",
    "StructuralModel",
    "PotentialEnergy",
    "MinimumEnergySolver",
    "StructuralSolution",
    "LineSystem",
    "build_line_system",
]
