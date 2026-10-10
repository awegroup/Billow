# Element library

| kernel | nodes | rot. DOF | key parameters |
|---|---|---|---|
| `CableKernel` | 2 | no | `rest_length`, `stiffness` |
| `PulleyKernel` | 3 | no | `rest_length`, `stiffness` |
| `TimoshenkoBeamKernel` | 2 | both | `rest_length`, `ea`/`ga_2`/`ga_3`/`gj`/`ei_2`/`ei_3`, `gamma_0`, `omega_0` |
| `InflatableBeamKernel` | 2 | both | tube law, diameter, pressure |
| `MembraneKernel` | 3 | no | `area`, `thickness`, `youngs_modulus`, `poisson_ratio`, `d0inv_*` |
| `PressureKernel` | 3 | no | `pressure` — one scalar per set |

## Cable

$U = \tfrac12 k\, s^2$ with $s$ the stretch. `tension_only` cuts compression,
which is C1; `slack_smoothing` rounds the corner when a smooth Hessian matters
more than the exact kink.

`line_tensions(positions, element_set)` reads the tension off the kernel's own
energy gradient rather than restating Hooke's law, so the slack cut it reports
is exactly the solver's: a slack line reads 0, a two-way element in compression
reads negative.

!!! warning "Feed `line_tensions` the positions the solver returned"
    A relaxed or interpolated shape is not in equilibrium, and on a stiff line a
    few millimetres of spurious stretch is hundreds of newtons.

## Pulley

Two arms sharing one stretch, so a frictionless sheave equalises the tension
either side of it. The `rest_length` parameter is the **whole rope**, both arms
together.

!!! note "This is where file formats disagree"
    Some line-system formats store the rope total on each arm; others split it
    across the two. Both are defensible, and a model built on the wrong reading
    puts every rope into artificial tension or compression. Any geometry adapter
    must state which convention it is reading — see
    [`build_line_system`](api/wireframe.md).

## Geometrically exact Timoshenko beam

One-point integration, following Simo & Vu-Quoc (1986). Reference strains
`gamma_0` and `omega_0` are stored, so a member that is curved as built is
stress-free as built.

!!! warning "Reference curvature is the quantity that breaks first on a real structure"
    `omega_0 = psi / L0` with `psi` a Rodrigues vector grows like
    $\tan(\theta/2)$ and is singular at $\theta = \pi$. A short element bridging
    a large frame change is numerically hostile long before it is physically
    wrong.

    Measured on a real kite: seeding each member's roll independently put the
    worst element at 173°, i.e. `omega_0 = 753 /m` on a 17 mm element. The tube
    section is isotropic in roll, so only `d1` is physically determined and the
    roll is free — transporting it by minimal rotation over the beam network,
    and letting the *long* member own each joint, gives 10.6 /m.

`initial_frames_from_polyline` and `rotations.minimal_rotation` are the tools
for that.

## Inflatable tube

The ASKITE tube fits, **integrated into a strain energy**.

The distinction matters. Applying those fits as *secant* stiffnesses —
recomputing `EI` and `GJ` from the current deflection and twist each iteration —
is consistent inside a Newton residual, but substituting a state-dependent
$EI(\kappa)$ into $\tfrac12 EI \kappa^2$ drops the $\mathrm{d}EI/\mathrm{d}\kappa$
terms, so the gradient stops being the internal moment. It is a force law, not a
potential.

The bending fit is published as tip load against normalised tip deflection of a
**one-metre** cantilever. Since $\kappa = 3v/L$ and $M = PL$ at the calibration
length, it becomes an intrinsic constitutive law:

$$
M(\kappa) = M_{\max}\left(1 - e^{-EI_0 \kappa / M_{\max}}\right),
\qquad EI_0 = N/3, \quad M_{\max} = D
$$

$$
W_b(\kappa) = M_{\max}\left[\,|\kappa| - k_0\left(1 - e^{-|\kappa|/k_0}\right)\right],
\qquad k_0 = M_{\max}/EI_0
$$

quadratic near zero, saturating at $M_{\max}$. Torsion is already intrinsic and
integrates directly to

$$
W_t = c_1\left[\omega \arctan(c_2 \omega) - \frac{\ln(1 + c_2^2\omega^2)}{2c_2}\right],
\qquad GJ_0 = c_1 c_2 .
$$

Recasting also removes a length inconsistency: inferring $EI = P/(3v)$ is the
true $EI$ only for a one-metre element, because $v$ is already normalised by
element length. A moment–curvature law is length-independent.

!!! success "Verified by pure end-moment solves"
    The exact answer is a constant-curvature arc at whatever curvature the fit
    prescribes — exact at any deflection, unlike the tip-load form, whose own
    inversion assumes linear cantilever theory. Worst error **0.06% in bending,
    1.5% in torsion**.

!!! note "Collapse is reported, not enforced"
    A dropping post-collapse moment would make the energy fall with curvature —
    an unbounded mechanism that minimisation would simply run away from.
    `inflatable_beam_state` returns `utilisation` (curvature over collapse
    curvature) and a `collapsed` flag, so leaving the calibrated range is
    visible rather than silent.

    Axial and shear stiffness are not covered by the fits and must be supplied
    from tube geometry.

## Wrinkling membrane

Constant-strain triangles on Pipkin's relaxed energy — the quasiconvex envelope
of the taut law. The minimiser resolves each element into one of three regimes,
reported by `membrane_regimes` as `SLACK`, `WRINKLED` or `TAUT`.

!!! danger "Fabric needs a residual slack stiffness"
    A fully slack region stores exactly zero energy, so its Hessian block is
    exactly zero and IPOPT fails with `Error_In_Step_Computation`.

    `MembraneKernel.slack_stiffness_ratio` (default `1e-4`) blends a small
    fraction of the unrelaxed law back in,
    $U = (1-r)\,U_{\text{relaxed}} + r\,U_{\text{taut}}$. The taut region is
    untouched, because the two coincide there.

    **Do not set it to zero on a canopy that can go slack.**

## Uniform pressure

A ram-air cell, a single skin pinned along its edge, or a tube modelled with
its own fabric is loaded by a uniform gauge pressure $p$ inside the same
minimisation. The work of a uniform pressure is $p$ times the swept volume, so
its potential is

$$
U = -p\,V, \qquad V = \sum_t \tfrac16\, \mathbf{x}_1\cdot(\mathbf{x}_2\times\mathbf{x}_3),
$$

the enclosed volume as a sum of signed cone volumes from the origin, one
three-node translation-only element per triangle. For a node whose fan of
triangles is closed, $\partial V/\partial \mathbf{x}_i$ is the vector area
of the ring polygon around it, $\tfrac13 \sum_t A_t \mathbf{n}_t$ — exactly
the force a lumped $p A \mathbf{n}/3$ scheme applies, at every interior node.
Positive $p$ inflates an **outward-wound** surface; `build_pressure_elements`
checks that the surface is closed, consistently wound and encloses a positive
volume, and names the first offending edges when it is not.

`pressure_forces(positions, element_set)` reads the nodal pressure forces off
the kernel's own gradient, and on a closed surface equals
`lumped_pressure_forces` to roundoff, with zero resultant and zero moment.

!!! note "Open surfaces: pin the rim"
    The cone volumes depend on where the origin is; their sum over a closed
    surface does not, and neither does the gradient at any node whose fan is
    closed. An open surface is therefore also a genuine potential provided every
    rim node is pinned — only the rim carries origin-dependent terms, and a
    pinned node's gradient never enters the equilibrium. `closed=False` relaxes
    the closure check; pin `surface_boundary_nodes(triangles)`.

!!! danger "One scalar per set — a per-triangle pressure is not a potential"
    The exactness at interior nodes holds only for a pressure that is uniform
    over the node's fan. With a per-triangle $p_t$ the sum $-\sum_t p_t V_t$
    is no longer translation-invariant (measured: 1.3e-13 N change under a
    translation for uniform $p$, 20.9 N for a per-triangle field), so its
    gradient is not a pressure force at all. `pressure` therefore stays a
    parameter column — ramp it on one solver build with
    `ElementSet.with_param_column("pressure", ...)` — but the builder takes one
    scalar, and `pressure_forces` refuses a column that is not constant over
    each connected piece of the surface. Nothing inside the solve checks it.

    Several cells at different pressures are several sets, one closed surface
    each. A rib shared by two cells appears in both with opposite windings and
    carries the net $p_\text{left} - p_\text{right}$; at equal pressure an
    internal rib carries nothing and belongs in no pressure set at all.

!!! danger "Never route an aerodynamic pressure distribution through this kernel"
    A flow-dependent $\Delta p$ over a single skin is non-uniform and has no
    potential. The isolation-consistent route is
    `lumped_pressure_forces(positions, triangles, pressure_per_triangle)`, a
    NumPy helper that lumps $p_t A_t \mathbf{n}_t / 3$ at the nodes, applied by
    the adapter as a dead load lagged by one coupling step — the same pattern
    as drag in `examples/run_quasi_steady_tether.py`. For uniform $p$ the lagged
    iteration and the potential share their fixed point (`test_pressure.py`).

Three more things to carry into a solve:

- **The mirrored half of a symmetric canopy must be wound in reversed order.**
  The triple product is odd under a reflection, so the "even in each strain"
  argument of the [symmetry constraints](formulation.md#symmetry) does not
  apply; the orientation check catches a half copied without the flip.
- **Pressure is not in the solver's load scaling.** The acceptance tolerance
  scales with the external `forces` only, so a pressure-only solve is judged
  at the absolute `force_tolerance` against nodal pressure forces of order
  $pA/n_\text{nodes}$. Pass a `force_tolerance` scaled to that — especially
  with `move_limit`, where the residual alone decides.
- **`StructuralSolution.strain_energy` includes $-pV$.** It is the total
  element energy; an inflated balloon reports a large negative number.

The SVK membrane and its relaxed form both grow like $\lambda^4$ in a uniform
stretch while $pV$ grows like $\lambda^3$, so the energy of an inflated balloon
is bounded below and a one-shot solve is well posed. The limit-point
instability of a neo-Hookean balloon is real physics this constitutive law
cannot show. A fixed-mass gas law, $-p_0 V_0 \ln(V/V_0)$, is a function of the
global volume rather than of one element and is not provided.

## Adding a kernel

A kernel is a pure function of one element's DOF, its frames and its parameter
row. It must:

- close over nothing element-specific — everything arrives through `params`;
- build only SX, so it can be compiled once and mapped;
- be **even in each strain component**, if the model it goes into will ever use
  the mirror-symmetry constraints (see [Formulation](formulation.md#symmetry))
  — or, like the pressure kernel, be odd in a way the mesh can absorb by
  reversing the mirrored winding;
- declare `nodes_per_element`, `param_names` and whether it needs rotational
  DOF.

Then check it against a closed-form solution, not against stored output of this
code.
