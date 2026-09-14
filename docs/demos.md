# Demonstration cases

Each case is a figure you can regenerate. They write into `results/demo/`.

```sh
pip install -e .[examples]
python examples/run_demo_cases.py --case wrinkling
```

Run without `--case` for all of them.

## `wrinkling` — fabric in a frame smaller than itself

A panel clamped into a frame 6% smaller than the panel. This is the case that
justifies the whole formulation.

| law | iterations | result |
|---|---|---|
| relaxed (Pipkin) | **31** | a smooth billow |
| unrelaxed | 211 | mesh-scale crumple |

The unrelaxed energy has no minimum in the compressive directions, so the solver
resolves the compression into whatever the mesh will let it: the answer is a
property of the discretisation, not of the fabric. The relaxed energy is the
quasiconvex envelope, so the minimiser lands on the wrinkled state directly and
the shape is mesh-independent.

## `cantilever` — large-deflection beam

A beam bent well past the small-rotation regime, to show the finite-rotation
handling at work in a case with a known answer. Compare with the `elastica`
benchmark in [Validation](validation.md), which measures the same physics
against the exact solution.

## `sail` — batten-stiffened sail

Membranes and beams in one model: fabric that wrinkles, carried by battens that
bend. This is the smallest case where the mixed-DOF layout matters — the batten
nodes carry six DOF and the fabric nodes three.

## `canopy_model` — spring net versus wrinkling membrane

The same canopy, once as a net of tension-only springs and once as wrinkling
membrane triangles, under the same load.

A spring net has no notion of a stress state: it cannot be *wrinkled*, only
slack or taut, member by member. Where the two agree and where they do not is
the substance of the canopy comparison in the
[technical document](billow.pdf).

## `scaling` — cost against mesh size

Build and solve time against triangle count, which is the measurement behind the
[assembly argument](formulation.md#assembly-one-mapped-kernel-per-element-type):

| triangles | DOF | build | solve | IPOPT iterations |
|---|---|---|---|---|
| 128 | 243 | 0.07 s | 0.05 s | 6 |
| 1800 | 2883 | 0.67 s | 1.25 s | 9 |
| 5832 | 9075 | 3.0 s | 6.1 s | 9 |

The iteration count barely moves. That is the signature of a well-posed
minimisation: refining the mesh buys resolution, not difficulty.

## `quasi_steady_tether` — a flying tether, and the catenary it rests on

```sh
python examples/run_quasi_steady_tether.py --case catenary
python examples/run_quasi_steady_tether.py --case circular
```

The two quasi-steady examples of
[Tethers.jl](https://github.com/ufechner7/Tethers.jl), with its default tether
and its load model: a kite fixed at [100, 100, 800] m against the analytic
catenary, and a kite flying one revolution of a 10° cone at 500 m while the
tether is re-solved at every 0.02 s sample from the previous shape.

The interesting part is where each quasi-steady load lives, because Billow
minimises a potential and only a load that *has* one can be exact inside the
solve:

| load | representation | exact? |
|---|---|---|
| weight | dead nodal force | yes |
| centrifugal term, `-m ω×(ω×r)` | one-node element kernel, `U = -½ m |ω×r|²`, with `ω` a live parameter | yes |
| segment drag, `c |vₙ| vₙ` | dead load evaluated on the previous step's shape | lagged one step |

Drag is velocity-dependent and non-conservative, so no potential has it as
gradient, and it cannot be folded into the energy. Applied one step lagged,
with no inner iteration, it costs 3e-3 N on a 57 N kite force and 1.1 mm on
the nodes over the sweep, measured against a damped Newton solve of the force
balance with the drag inside the residual. That Newton solve is the wrong tool
for a taut chain: a 2 m transverse step on a 26 m segment adds 8 cm of stretch,
so from a cold start its residual-norm line search collapses to a hundredth of
a step, where IPOPT, with the energy as merit function, takes four iterations.

Two things a port of this model has to get right, both found the hard way:

* the cables are **two-way** springs. Tethers.jl lays each segment along
  the accumulated force, `l = (|F|/EA + 1) Ls`, so its segments are taut by
  construction and the two laws coincide on that branch, which a hanging
  tether never leaves. The slack cut is dropped because on a catenary seed
  every curved segment's chord is shorter than its arc, so the chain would
  start with zero stiffness and IPOPT stops with
  `Error_In_Step_Computation`;
* the seed is the analytic catenary spaced by **arc length**, not horizontal
  distance.

`tests/test_catenary.py` is the closed-form check behind it: a hanging chain of
cables converges onto the analytic catenary at O(h²), and its segment tensions
onto the catenary tension at the segment midpoints.

One warm step costs 4–7 ms here against 9 µs for the Julia shooting solver,
which closes shape and loads in one three-unknown root-find. The port is not a
faster tether; it is the same tether inside the formulation that also carries
the bridle, the tube frame and the canopy.

## Cost studies

```sh
python examples/run_panel_cost.py        # one panel, mesh refinement
python examples/run_full_kite_cost.py    # a whole kite, node count
python examples/run_cost_comparison.py   # against the external code, if installed
```

These write markdown tables rather than figures, and are what the cost numbers
quoted elsewhere in the documentation come from.
