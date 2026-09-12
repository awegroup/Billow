# Formulation

## Total potential energy

Billow finds the configuration $\mathbf{x}$ that minimises

$$
\Pi(\mathbf{x}) \;=\; \sum_e W_e(\mathbf{x}) \;-\; \mathbf{f}^{\mathsf T}\mathbf{x}
\;-\; \sum_n \mathbf{M}_n \cdot \boldsymbol{\theta}_n
$$

the sum of every element's strain energy less the work of the applied loads.
Stationarity of $\Pi$ *is* static equilibrium, so the gradient of the objective
is the out-of-balance force and needs no separate assembly. That is why
`internal_load` and `tangent_stiffness` come free, and why the residual Billow
reports is the residual of the thing it minimised.

!!! note "Loads are dead"
    Loads are frozen per solve. That matches a staggered coupling loop, where
    the flow solver supplies a load field and the structure answers with a
    shape. A true follower pressure would not be conservative and could not be
    posed as a potential at all — worth knowing before anyone tries to fold
    aerodynamics into the same minimisation.

## Degrees of freedom

Three translational DOF on every node, and three rotational DOF **only on nodes
a kernel asks for them**. Cable, pulley and membrane nodes stay at three. A
canopy of 10 000 fabric nodes therefore costs 30 000 DOF, not 60 000, whatever
beams are attached elsewhere.

## Finite rotations: the Cayley chart

Rotations are incremental Rodrigues (Cayley) vectors:

$$
\mathbf{R} = \operatorname{cay}(\boldsymbol{\psi})\,\mathbf{R}_{\text{ref}},
\qquad
\operatorname{cay}(\boldsymbol{\psi}) = \mathbf{I} +
\frac{4}{4+\|\boldsymbol{\psi}\|^2}
\left( 2\,\widehat{\boldsymbol{\psi}} + \widehat{\boldsymbol{\psi}}^2 \right)
$$

with $\mathbf{R}_{\text{ref}}$ stored in the state and every solve starting from
$\boldsymbol{\psi} = \mathbf{0}$. Two choices there, both load-bearing:

**Cayley rather than the exponential map.** Every resulting formula is
*rational*. No $\sin\phi/\phi$ with a removable singularity, so no `if_else`
branch and no kink in the Hessian IPOPT differentiates.

**Incremental rather than total.** The map is singular at $\phi = \pi$. The
solver absorbs each solved increment into $\mathbf{R}_{\text{ref}}$ and starts
the next solve from zero, so a node can turn arbitrarily far across a load ramp
while every individual solve stays well inside the chart.

!!! warning "An applied moment is not conjugate to $\boldsymbol{\psi}$"
    A moment is work-conjugate to the **exponential-map** rotation vector, not
    to the Rodrigues vector. The external work term converts (see
    `rotations.rotation_vector`); using $-\mathbf{M}\cdot\boldsymbol{\psi}$
    directly made the answer depend on the load stepping.

    Separately, a constant "dead" moment in 3-D is genuinely non-conservative
    and has no potential at all, so energy minimisation can only represent the
    fixed-axis case. Applied moments also need load stepping — four steps is
    enough for a full roll-up.

    Nodal **forces**, which is what an aerodynamic coupling applies, carry none
    of these caveats and are exact in one solve.

## Assembly: one mapped kernel per element *type*

This is the entire scalability argument.

Each `ElementKernel.energy` is compiled once to a small SX `casadi.Function`
over a **single** element, then evaluated across the whole set with
`Function.map` on a gathered index matrix. The objective graph therefore holds
one node per element *type*, not per element.

!!! danger "Never reintroduce a Python loop over elements that builds MX"
    At tens of springs a per-element loop is free. At tens of thousands of
    fabric triangles the graph build — and especially `gradient`/`hessian` on it
    — becomes the bottleneck long before IPOPT does.

Measured on a flat clamped canopy (`examples/run_demo_cases.py --case scaling`):

| triangles | DOF | build | solve | IPOPT iterations |
|---|---|---|---|---|
| 128 | 243 | 0.07 s | 0.05 s | 6 |
| 1800 | 2883 | 0.67 s | 1.25 s | 9 |
| 5832 | 9075 | 3.0 s | 6.1 s | 9 |

Iteration count is essentially mesh independent; the cost is in the sparse
linear algebra, where it belongs.

`tests/test_energy.py` checks the mapped assembly against a naive per-element
sum. That is the key test of this design.

## Everything numeric is a parameter

Reference frames, every element parameter column (rest lengths, stiffnesses,
section properties), loads, and the anchor are all NLP **parameters**, never
baked constants.

One `MinimumEnergySolver` build therefore serves a whole sweep: actuation and
stiffness ramps change parameters, never topology. `StructuralModel.replaced`
and `ElementSet.with_param_column` are the supported way to do that, and
[`LineSystem`](api/wireframe.md) wraps them for line systems.

## Kernels are pure

No element loops, no global indexing, no closing over per-element data.
Everything element-specific arrives through `params`. A kernel that closes over
anything cannot be mapped, which breaks the assembly argument above.

## Single source for every formula

`rotations.py` holds the SO(3) maps and is written against an `xp` namespace, so
NumPy and CasADi share one implementation. `beam_strains` and `green_strain` are
likewise called by both the kernels and the NumPy setup and diagnostic paths.

A strain measure written twice is a strain measure that will disagree with
itself. Do not restate one.

## Convergence

`converged` is the **physical** verdict — a force balance within
`force_tolerance` — with IPOPT's own verdict preserved as `ipopt_success` and
`status`.

With `equalities` active (for example the mirror-symmetry constraints in
`symmetry.py`), IPOPT solves the constrained problem, whose optimum may be held
by a reaction, and the reported residual **includes that reaction**. It is zero
only when the constrained state is an equilibrium of the free problem too —
which makes the residual a check on the constraints. A symmetric-constrained
solve of a model that is not really symmetric shows the defect as a residual.

## Symmetry

`symmetry.py` provides mirror pairing and the linear equalities that impose it:
$\mathbf{x}_p = \mathbf{M}\mathbf{x}_i$ for positions and
$\boldsymbol{\psi}_p = -\mathbf{M}\boldsymbol{\psi}_i$ for the rotation
increments, which are pseudovectors.

!!! warning "A mirror-symmetric structure needs mirror-consistent frames"
    The roll of a beam frame is a gauge *along a member*, but not node by node:
    the element measures the relative rotation of its two end frames, so rolls
    that differ between neighbours change the energy.

    Minimal-rotation transport over a network that mixes a member crossing the
    mirror plane (whose tangent maps as $-\mathbf{M}\mathbf{t}$) with members
    beside it ($+\mathbf{M}\mathbf{t}$) is **not** mirror-equivariant. On a real
    kite the far half arrived rolled by up to 60° with every node position
    symmetric to 4e-15 m, and the structure solved to one deterministic
    asymmetric shape.

    `symmetry.mirror_frames` fixes it with one director-sign matrix
    $\mathbf{S}$ for every node, $\mathbf{R}_p = \mathbf{M}\mathbf{R}\mathbf{S}$.
    Each element's strains then map to a sign-flipped copy of its mirror
    element's, and every kernel here is even in each strain component — **check
    any new kernel for that evenness before relying on it.**
