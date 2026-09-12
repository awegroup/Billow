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

## Cost studies

```sh
python examples/run_panel_cost.py        # one panel, mesh refinement
python examples/run_full_kite_cost.py    # a whole kite, node count
python examples/run_cost_comparison.py   # against the external code, if installed
```

These write markdown tables rather than figures, and are what the cost numbers
quoted elsewhere in the documentation come from.
