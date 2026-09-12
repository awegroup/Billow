# Validation

Every benchmark below is measured against a reference **outside this code** — an
exact solution, a boundary-value problem solved to 1e-12, published values, or a
physical measurement. None is a stored regression of Billow's own output.

```sh
python validation/run_validation_benchmarks.py            # all beam cases
python validation/run_validation_benchmarks.py --case rollup --show
python validation/run_hanging_validation.py               # the measured kite
```

## Large-rotation beam benchmarks

| benchmark | reference | Billow |
|---|---|---|
| Euler elastica, $\alpha$ up to 10 | exact BVP solution | 1e-4 … 1.1e-3 $L$ at 40 elements |
| Roll-up under $M = 2\pi EI/L$ | exact closed circle | 1.13e-1 / 3.16e-2 / 8.14e-3 $L$ at 10/20/40 elements — **O(h²)** |
| Bathe & Bolourchi 45° bend | published tip displacement | within the spread between published values, 0.2% on the magnitude |
| Inflatable tube, pure end moment | the fitted moment–curvature law | 0.06% bending, 1.5% torsion |

The roll-up case is the sharpest of the three: the exact answer is a closed
circle, so the error is the gap between the tip and the root and there is
nothing to look up. Convergence is second order and **independent of the load
stepping**, which is the property that says the rotation handling is right
rather than merely tuned.

## Comparison with an external code

Installing the optional comparison dependency adds a `kite_fem` / `pyfe3d`
column to the beam tables:

```sh
pip install -e .[compare]
```

Its absence removes a column, never an assertion — every reference above stands
on its own.

Two things that comparison required, both worth knowing before running one:

!!! warning "An absolute identity stiffness is a kite-scale tuning constant"
    `FEM_structure.solve` defaults to `I_stiffness=25`, an **absolute** N/m
    added to every DOF of the tangent. On these slender benchmark beams
    $EI/L^3$ is 10 N/m, so the default is larger than the structure and the
    solve stalls at a residual of 20 N. The benchmarks run it at zero — the
    setting that works, not the shipped default.

!!! note "Matching properties is not the same as matching formulations"
    `kite_fem`'s beam matches linear theory to 0.02% below about one degree of
    rotation, so the property matching is right. Its error is purely
    rotation-driven and **mesh independent**, which places it in the
    formulation rather than in the discretisation — it grows with refinement on
    the roll-up case (5.9e-2 → 1.71e-1 → 2.32e-1 $L$ at 10/20/40 elements).

## Element-level tests

| file | what it asserts |
|---|---|
| `test_rotations.py` | Cayley round trips and the half-angle identity |
| `test_cable.py` | Hooke parity, the slack cut, pulley tension equalisation, `line_tensions` |
| `test_wireframe.py` | line-system statics, state carried across solves, actuation without a rebuild |
| `test_beam.py` | objectivity, O(h²) convergence to Timoshenko, no shear locking, torsion vs `GJ`, pre-curved members stress-free |
| `test_membrane.py` | analytic SVK, the three wrinkling branches, flat-plate inflation, mesh convergence, scale |
| `test_energy.py` | **mapped assembly vs a naive per-element sum** — the key test of the design — and `internal_load` vs finite differences |
| `test_inflatable.py` | the energy integrates the fitted moment; pure end-moment solves land on the fitted curve; collapse reporting |
| `test_symmetry.py` | pairing and equalities; frames transported through junctions break mirror symmetry, `mirror_frames` restores it to roundoff |
| `test_benchmarks.py` | roll-up vs the exact circle, Bathe & Bolourchi 45° bend |
| `test_hanging_kite.py` | the measured-kite model builds, and the length extractor reproduces the reference results exactly |

Tests assert against closed-form solutions and convergence orders, never against
stored solver output. A test that pins today's numbers cannot tell you the
physics changed.

## Validation against a measured kite

The TU Delft V3 kite hangs upside down in a hangar from a bar, with no
aerodynamic loading, and the deformed shape is measured by stereoscopic camera.
Under self-weight alone the span falls from **8.305 m as built to 4.867 m** — a
41% closure, with every trailing-edge segment shortening 26–43%. That closure is
the deformation the dataset really tests.

Ten load cases combine two inflation pressures with a mid-span point load and a
tip-spreading load. Twelve lengths are measured off each deformed shape.

The model is rebuilt from the reference solver's own assembled model, so nodes,
connectivity, rest lengths, stiffnesses and tube diameters transfer with no
re-interpretation. **Only the canopy changes:**

| | canopy |
|---|---|
| reference | 721 noncompressive springs on a 29 × 8 grid |
| Billow | 392 wrinkling CST membrane triangles on the same grid |

Everything else — 46 bridle cables, 14 pulleys, 98 inflatable tube beams — is
element-for-element the same. That is deliberate: it makes the difference
between the two answers attributable to the canopy formulation alone.

!!! warning "Defects found in the stored reference model"
    Working through this case surfaced three problems in the published model,
    all documented in
    [`validation/data/hanging_validation/README.md`](https://github.com/awegroup/Billow/blob/main/validation/data/hanging_validation/README.md)
    and at length in the [technical document](billow.pdf):

    - **The anchor is connected to nothing.** Node 0 is flagged fixed but
      appears in no spring, pulley or beam, and seven further bridle nodes are
      orphaned with it. Anything solving this file must supply a suspension.
      The thesis explains why: the bridle was adapted to fit the hangar height,
      and the kite hangs from a **1650 mm bar**, not a point.
    - **Bridle lengths disagree with the thesis.**
    - **The reference beams converge far softer than their own law.**

See the [technical document](billow.pdf) for the case-by-case geometry, the
centre-load cases that neither code reproduces, and an honest account of how
weakly twelve lengths constrain a model.
