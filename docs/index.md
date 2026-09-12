# Billow

**Minimum-energy structural modelling of soft, inflatable structures.**

Billow solves for the static equilibrium of a soft structure — a bridle of
tension-only lines, an inflatable tube frame, a canopy of fabric that wrinkles
rather than buckles — by minimising one total potential energy and handing it to
IPOPT.

## Why energy minimisation

It is not a solver preference. A slack line and a wrinkled panel are exactly the
states where a residual-form Newton solve has to fight a near-singular tangent:
the structure has directions in which it carries no stiffness at all, so the
tangent matrix is singular and the Newton step is undefined without
regularisation.

Posed instead as a minimisation over the *relaxed* (quasiconvex) energy, those
states are simply where the minimum is. The minimiser lands on them directly.

!!! example "Measured on a panel clamped into a frame 6% smaller than itself"
    The relaxed law converges in **31 iterations** to a smooth billow.
    The unrelaxed one takes **211** and buckles into mesh-scale crumple.

Wrinkling is therefore the reason the formulation is right here, not a
complication it has to survive.

## Two fidelities, one formulation

| | elements | DOF | typical use |
|---|---|---|---|
| **wireframe** | cables, tension-only lines, frictionless pulleys | 3 per node | the bridle and line system on its own |
| **full** | the above + geometrically exact Timoshenko beams, inflatable-tube laws, wrinkling CST membranes | 3 per node, +3 on beam nodes only | the whole kite |

Both are the same pipeline:

```
nodes + element sets  ->  StructuralModel
StructuralModel       ->  MinimumEnergySolver   (compiled once)
solver.solve(forces)  ->  StructuralSolution    (state, residual, energy)
```

The simplified model is a *subset* of the full one, not a separate code path.
There is one cable force law in this package and both fidelities call it, so
they cannot drift apart.

## Isolation

Billow imports **only NumPy and CasADi**. It knows nothing about aerodynamics,
coupling loops, or any particular file format.

That boundary is the first rule of the package. It is what lets the element
physics be validated against closed-form solutions before any of it touches a
coupled run — and what keeps a coupling bug from being mistaken for a physics
bug. Coupling Billow to a flow solver belongs in an adapter on the other side of
that boundary; see [Coupling Billow](coupling.md).

## Where to go next

<div class="grid cards" markdown>

- **[Getting started](getting-started.md)** — install, first solve, reading a solution
- **[Formulation](formulation.md)** — the energy, the rotation chart, the assembly
- **[Element library](elements.md)** — what each kernel is and what it needs
- **[Validation](validation.md)** — external references, and what they measure
- **[Demonstration cases](demos.md)** — the runnable figures
- **[Full technical document](billow.pdf)** — everything above at length

</div>
