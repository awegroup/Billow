# Billow

**Minimum-energy structural modelling of soft, inflatable structures.**

[![tests](https://github.com/awegroup/Billow/actions/workflows/tests.yml/badge.svg)](https://github.com/awegroup/Billow/actions/workflows/tests.yml)
[![docs](https://img.shields.io/badge/docs-awegroup.github.io%2FBillow-2563eb?style=flat&logo=githubpages&logoColor=white)](https://awegroup.github.io/Billow/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Billow solves for the static equilibrium of a soft structure — a bridle of
tension-only lines, an inflatable tube frame, a canopy of fabric that wrinkles
rather than buckles — by **minimising one total potential energy** and handing
it to IPOPT.

That is not a solver preference. A slack line and a wrinkled panel are exactly
the states where a residual-form Newton solve has to fight a near-singular
tangent; posed as a minimisation over the *relaxed* (quasiconvex) energy, the
minimiser lands on them directly. On a panel clamped into a frame 6% smaller
than itself, the relaxed law converges in **31 iterations** to a smooth billow;
the unrelaxed one takes **211** and buckles into mesh-scale crumple.

<!-- TODO: hero figure — the wrinkling demo, results/demo/wrinkling.png -->

## Two fidelities, one formulation

|  | elements | degrees of freedom | typical use |
|---|---|---|---|
| **wireframe** | cables, tension-only lines, frictionless pulleys | 3 per node | the bridle and line system on its own |
| **full** | the above + geometrically exact Timoshenko beams, inflatable-tube laws, wrinkling CST membranes | 3 per node, +3 on beam nodes only | the whole kite |

Both are the same `StructuralModel` → `MinimumEnergySolver` → `StructuralSolution`
pipeline, so the simplified model is a *subset* of the full one rather than a
separate code path. The force laws cannot drift apart because there is only one
of each.

## Install

```sh
pip install git+https://github.com/awegroup/Billow.git
```

The runtime dependencies are **NumPy and CasADi**, and that is deliberate — see
[Isolation](#isolation). For the demonstration and validation scripts:

```sh
pip install "billow[examples] @ git+https://github.com/awegroup/Billow.git"
```

Development install:

```sh
git clone https://github.com/awegroup/Billow.git
cd Billow
pip install -e .[dev,examples]
pytest
```

## Sixty seconds

```python
import numpy as np
from billow import MinimumEnergySolver, StructuralModel
from billow.elements import build_cable_elements

# Two cables from fixed anchors to a free node, loaded downward.
nodes = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
cables = build_cable_elements(
    connectivity=[[0, 2], [1, 2]], rest_lengths=[1.0, 1.0], stiffness=[1e4, 1e4]
)

model = StructuralModel(nodes, [cables], fixed_translation_nodes=[0, 1])

forces = np.zeros((3, 3))
forces[2] = [0.0, 0.0, -500.0]

solution = MinimumEnergySolver(model).solve(forces)
print(solution.converged, solution.state.positions[2])
```

`solution` carries the equilibrium state, the force residual that decides
convergence, the internal loads and the energy. `converged` is the **physical**
verdict — a force balance — with IPOPT's own verdict preserved separately as
`ipopt_success`; IPOPT routinely reports `Search_Direction_Becomes_Too_Small` on
stiff structures whose answer is already exact.

## Element library

| kernel | nodes | rot. DOF | notes |
|---|---|---|---|
| `CableKernel` | 2 | no | `tension_only` cuts compression; `slack_smoothing` rounds the corner |
| `PulleyKernel` | 3 | no | two arms sharing one stretch; rest length is the **whole rope** |
| `TimoshenkoBeamKernel` | 2 | both | geometrically exact; reference strains make curved members stress-free |
| `InflatableBeamKernel` | 2 | both | ASKITE tube fits **integrated into a strain energy**, not applied as a secant stiffness |
| `MembraneKernel` | 3 | no | Pipkin relaxed energy — slack / wrinkled / taut resolved by the minimiser |

## Validation

Every benchmark is measured against a reference **outside** this code — an exact
solution, a boundary-value problem solved to 1e-12, or published values.

```sh
python validation/run_validation_benchmarks.py      # elastica, roll-up, 45° bend, inflatable tube
python validation/run_hanging_validation.py         # the measured hanging V3 kite
```

| benchmark | reference | Billow |
|---|---|---|
| Euler elastica, α up to 10 | exact BVP solution | 1e-4 … 1.1e-3 L at 40 elements |
| Roll-up under M = 2πEI/L | exact closed circle | 1.13e-1 / 3.16e-2 / 8.14e-3 L at 10/20/40 elements — **O(h²)** |
| Bathe & Bolourchi 45° bend | published tip displacement | within the spread between published values, 0.2% on the magnitude |
| Inflatable tube, pure end moment | the fitted moment–curvature law | 0.06% in bending, 1.5% in torsion |
| Hanging V3 kite | stereoscopic marker measurement | see [`validation/data/hanging_validation/README.md`](validation/data/hanging_validation/README.md) |

Installing the optional comparison code adds a `kite_fem` column to the beam
tables (`pip install -e .[compare]`); its absence removes a column, never an
assertion.

## Demonstration cases

```sh
python examples/run_demo_cases.py --case wrinkling    # fabric in a frame smaller than itself
python examples/run_demo_cases.py --case cantilever   # large-deflection beam
python examples/run_demo_cases.py --case sail         # batten-stiffened sail
python examples/run_demo_cases.py --case canopy_model # spring net vs. wrinkling membrane
python examples/run_demo_cases.py --case scaling      # cost against mesh size
```

## Why it scales

One mapped kernel per element **type**, never per element. Each
`ElementKernel.energy` is compiled once to a small SX `casadi.Function` over a
single element and evaluated across the whole set with `Function.map` on a
gathered index matrix, so the objective graph holds one node per element type
and the build cost does not grow with the mesh.

| triangles | DOF | build | solve | IPOPT iterations |
|---|---|---|---|---|
| 128 | 243 | 0.07 s | 0.05 s | 6 |
| 1800 | 2883 | 0.67 s | 1.25 s | 9 |
| 5832 | 9075 | 3.0 s | 6.1 s | 9 |

Iteration count is essentially mesh independent; the cost is in the sparse
linear algebra, where it belongs.

## Isolation

Billow imports **only NumPy and CasADi**. It knows nothing about aerodynamics,
coupling loops, or any particular kite file format. That boundary is the first
rule of the package, and it is what lets the element physics be validated
against closed-form solutions before any of it touches a coupled run.

Coupling it to a flow solver belongs in an adapter on the other side of the
boundary. The reference one is
[AWETrim](https://github.com/awegroup/AWETrim)'s `aerostructural/` package,
which drives Billow against a vortex-step-method quasi-steady trim.

## Documentation

- **[Full technical document](docs/billow.pdf)** — formulation, element library,
  validation, demonstration cases, the measured-kite study and the canopy mesh
  study. Source: [`docs/billow.tex`](docs/billow.tex).
- **[Online documentation](https://awegroup.github.io/Billow/)**
- **[`AGENTS.md`](AGENTS.md)** — design decisions, conventions to preserve, and
  the open items. Read this before changing element physics.

## References

The governing equations trace back to:

- **Beam:** Simo & Vu-Quoc (1986) *Comput. Methods Appl. Mech. Engrg.* **58**, 79–116.
- **Rotation parameterisation:** Ibrahimbegović (1997) *Comput. Methods Appl. Mech. Engrg.* **149**, 49–71.
- **Membrane wrinkling:** Pipkin (1986) *IMA J. Appl. Math.* **36**, 85–99;
  Roddeman et al. (1987) *J. Appl. Mech.* **54**, 884–892.
- **Benchmarks:** Bathe & Bolourchi (1979) *Int. J. Numer. Meth. Engng.* **14**, 961–986;
  Bisshopp & Drucker (1945) *Q. Appl. Math.* **3**, 272–275.
- **Aerostructural context:** Cayon, Gaunaa & Schmehl (2023) *Energies* **16**, 3061.

## Licence

Apache-2.0 — see [LICENSE](LICENSE). Portions derive from
[ASKITE](https://github.com/awegroup/ASKITE) (MIT) and the validation data is
vendored from [kite_fem](https://github.com/awegroup/kite_fem) (MIT); see
[NOTICE](NOTICE).
