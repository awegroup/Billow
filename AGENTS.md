# Billow — Agent Context

## What this repo is

Billow is a standalone structural solver for soft, inflatable structures:
cables, frictionless pulleys, geometrically exact Timoshenko beams, inflatable
tube constitutive laws and wrinkling membrane fabric, assembled into **one total
potential energy** and solved for static equilibrium with IPOPT.

Two fidelities, one formulation:

| | elements | DOF |
|---|---|---|
| **wireframe** (`wireframe.py`) | cables, tension-only lines, pulleys | 3 per node |
| **full** | the above + beams, inflatable tubes, wrinkling CST membranes | 3 per node, +3 on beam nodes only |

The wireframe model is a *subset* of the full one, not a second code path. There
is one cable force law in this package and both fidelities call it.

## The first rule: isolation

**This package imports only NumPy and CasADi.** No aerodynamics, no coupling
loop, no YAML schema, no file format. Adding a runtime dependency is a design
decision, not a convenience.

That boundary is what lets the element physics be validated against closed-form
solutions before any of it touches a coupled run, and what keeps a coupling bug
from being mistaken for a physics bug. Coupling belongs in an adapter on the
other side of it — the reference one is
[AWETrim](https://github.com/awegroup/AWETrim)'s `aerostructural/` package.

`plotting.py` is the one concession, and it is not a solver dependency:
matplotlib is imported lazily inside `set_plot_style`, so a headless install
never pays for it, and nothing under `src/billow/` outside that module imports
it.

## Layout

```
src/billow/
  __init__.py     StructuralModel, StructuralState, DofLayout, PotentialEnergy,
                  MinimumEnergySolver, StructuralSolution, LineSystem,
                  build_line_system
  rotations.py    SO(3) kernels over an `xp` namespace (NumPy or CasADi):
                  cayley, cayley_vector, half_vector, skew, axial,
                  orthonormalize, minimal_rotation, unit_vector,
                  frames_to_flat / flat_to_frames / unflatten_frame
                  (`minimal_rotation` lives here rather than in beam.py because
                   coupling adapters transport frames with it too)
  model.py        DofLayout (gather maps), StructuralState (positions + frames),
                  StructuralModel (elements + reference config + pinned DOF)
  energy.py       PotentialEnergy: mapped assembly, objective, internal_load,
                  tangent_stiffness, parameter packing
  solver.py       MinimumEnergySolver, StructuralSolution
                  (optional equalities=LinearEqualities: C X = 0 compiled into
                   the NLP as g, for constrained solves)
  symmetry.py     mirror symmetry: mirror_partners, mirror_equalities
                  (x_p = M x_i for positions, psi_p = -M psi_i for the rotation
                  increments, which are pseudovectors), LinearEqualities,
                  mirror_frames (R_p = M R diag(-1,1,1)), frame_mirror_mismatch
  wireframe.py    LineSystem, build_line_system, pulley_triplets -- the
                  simplified fidelity, plus state ownership across a sequence of
                  solves (positions persist; rest lengths and stiffnesses are
                  addressable by the CALLER's line index). Lines and elements
                  are NOT one-to-one: a pulley rope is one three-node element
                  carried at two arm indices, and tension-only and two-way lines
                  compile to different kernels, so `line_set` / `line_row` is
                  the map and every accessor speaks the caller's index.
  plotting.py     PALETTE, COLOR_CYCLE, REGIME_COLORS, set_plot_style --
                  figures only; matplotlib imported lazily
  elements/
    base.py       ElementKernel protocol, ElementSet, local DOF accessors
    cable.py      CableKernel, PulleyKernel + build_cable_elements /
                  build_pulley_elements, line_tensions (per-line tension from
                  the kernel's own energy gradient -- slack cut included; feed
                  it SOLVED positions)
    beam.py       TimoshenkoBeamKernel, BeamSection, beam_strains,
                  build_beam_elements, initial_frames_from_polyline
    inflatable.py InflatableTubeLaw, InflatableBeamKernel,
                  build_inflatable_beam_elements, inflatable_beam_state --
                  the ASKITE tube fits integrated into a strain energy
    membrane.py   MembraneKernel, green_strain, membrane_reference,
                  build_membrane_elements, membrane_regimes, SLACK/WRINKLED/TAUT

tests/
  test_rotations.py  Cayley round trips and half-angle identity
  test_cable.py      Hooke parity, slack cut, pulley tension equalisation,
                     line_tensions (taut / slack / compression / solved rope)
  test_wireframe.py  line-system statics, pulley pairing, state across solves,
                     actuation and stiffness ramps without a rebuild
  test_beam.py       objectivity, O(h^2) convergence to Timoshenko, no shear
                     locking, torsion vs GJ, pre-curved members stress-free
  test_membrane.py   fabric on its own -- analytic SVK, the three wrinkling
                     branches, flat-plate inflation, mesh convergence, scale
  test_energy.py     mapped assembly vs a naive per-element sum (the key test),
                     internal_load vs finite differences, DOF layout
  test_benchmarks.py roll-up vs the exact circle (O(h^2) + load-step
                     independence), Bathe and Bolourchi 45 degree bend
  test_inflatable.py energy integrates the fitted moment; pure end-moment
                     solves land on the fitted curve; collapse reporting
  test_symmetry.py   pairing and equalities; frames transported through strut
                     junctions break mirror symmetry, mirror_frames restores it
  test_hanging_kite.py  the measured-kite model builds; the length extractor
                     reproduces the reference results exactly

examples/           runnable figures and cost studies -> results/demo/
  run_demo_cases.py           wrinkling / canopy_model / cantilever / sail /
                              scaling
  run_panel_cost.py, run_full_kite_cost.py, run_cost_comparison.py

validation/         external references -> results/validation/, results/hanging/
  run_validation_benchmarks.py  elastica / rollup / bend45 / inflatable,
                                optionally against kite_fem with matched
                                properties
  hanging_kite.py               the measured TU Delft V3, rebuilt as a Billow
                                model from the reference assembled model
  run_hanging_validation.py, run_hanging_shape_comparison.py,
  run_hanging_stiffness_scan.py, plot_hanging_validation.py,
  make_hanging_report.py, run_mesh_requirement.py
  data/hanging_validation/      vendored measurement data (MIT, see its README)

docs/               mkdocs-material site + billow.tex, the full technical
                    document (formulation, elements, validation, demos, the
                    kite_fem comparison, the canopy mesh study)
```

## The four design decisions

**1. One mapped kernel per element TYPE, not per element.** Each
`ElementKernel.energy` is compiled once to a small SX `casadi.Function` over a
single element and evaluated across the whole set with `Function.map` on a
gathered index matrix. The objective graph therefore holds one node per element
*type*. This is the entire scalability argument — **never reintroduce a Python
loop over elements that builds MX.**

Measured (flat clamped canopy, `examples/run_demo_cases.py --case scaling`):

| triangles | DOF  | build | solve | IPOPT iters |
|-----------|------|-------|-------|-------------|
| 128       | 243  | 0.07 s| 0.05 s| 6  |
| 1800      | 2883 | 0.67 s| 1.25 s| 9  |
| 5832      | 9075 | 3.0 s | 6.1 s | 9  |

Iteration count is essentially mesh independent; cost is in the sparse linear
algebra, where it belongs.

**2. Rotations are incremental Rodrigues (Cayley) vectors.** `R = cayley(psi) R_ref`
with `R_ref` stored in the state and the solve always starting from `psi = 0`.
Cayley rather than the exponential map because every resulting formula is
*rational* — no `sin(phi)/phi` removable singularity, so no `if_else` branch and
no kink in the Hessian IPOPT differentiates. Incremental rather than total
because the map is singular at `phi = pi`: the solver absorbs each solved
increment into `R_ref` and starts the next solve from `psi = 0`, so a node can
turn arbitrarily far across a load ramp while every individual solve stays well
inside the chart. An applied moment is work-conjugate to the exponential-map
rotation vector, not to `psi`, so the external work term converts (see
`rotations.rotation_vector`); using `-M . psi` directly made the answer depend on
the load stepping.

**3. Rotational DOF only where a kernel asks for them.** Cable, pulley and
membrane nodes stay at three DOF. A canopy of 10 000 fabric nodes costs 30 000
DOF, not 60 000, whatever beams are attached elsewhere.

**4. Everything numeric is an NLP parameter.** Reference frames, every element
parameter column (rest lengths, stiffnesses, section properties), loads, anchor.
One `MinimumEnergySolver` build serves a whole sweep: actuation and stiffness
ramps change parameters, never topology. `StructuralModel.replaced` and
`ElementSet.with_param_column` are the supported way to do that, and
`LineSystem` wraps them for line systems.

## Validation

`validation/run_validation_benchmarks.py` runs the classical large-rotation
benchmarks. References come from outside this code; `kite_fem`/`pyfe3d` joins
the tables when the optional `[compare]` extra is installed, and its absence
removes a column, never an assertion.

| benchmark | reference | Billow | kite_fem |
|-----------|-----------|--------|----------|
| Euler elastica, alpha up to 10 | exact BVP solution | 1e-4 to 1.1e-3 L at 40 el | 1.6e-2 to 5e-1 L |
| Roll-up under M = 2 pi EI/L | exact closed circle | 1.13e-1 / 3.16e-2 / 8.14e-3 L at 10/20/40 el, O(h^2) | 5.9e-2 / 1.71e-1 / 2.32e-1 L, diverges with refinement |
| Bathe and Bolourchi 45 degree bend | published tip displacement | within the spread between published values, 0.2% on the magnitude | out-of-plane component 47% low |

Two things that comparison required, both worth knowing:

* `FEM_structure.solve` defaults to `I_stiffness=25`, an **absolute** N/m added
  to every DOF of the tangent. On these benchmark beams `EI/L^3` is 10 N/m, so
  the default is larger than the structure and the solve stalls at a residual of
  20 N. Benchmarks run it at zero. It is a kite-scale tuning constant, not a
  universal one.
* `kite_fem`'s beam matches linear theory to 0.02% below about one degree of
  rotation, so the property matching is right; its error is purely
  rotation-driven and **mesh independent**, which places it in the formulation
  rather than in the discretisation.

### The inflatable tube

`kite_fem` applies the ASKITE fits as *secant* stiffnesses, recomputing `EI` and
`GJ` from the current deflection and twist each iteration. That is a force law,
not a potential: substituting a state-dependent `EI(kappa)` into `1/2 EI kappa^2`
drops the `dEI/dkappa` terms, so the gradient stops being the internal moment.
`elements/inflatable.py` integrates them instead.

The bending fit is published as tip load against normalised tip deflection of a
**one-metre** cantilever. Since `kappa = 3 v / L` and `M = P L` at the
calibration length, it becomes an intrinsic constitutive law:

    M(kappa) = M_max (1 - exp(-EI_0 kappa / M_max)),  EI_0 = N/3,  M_max = D
    W_b(kappa) = M_max [ |kappa| - k0 (1 - exp(-|kappa|/k0)) ],  k0 = M_max/EI_0

quadratic near zero, saturating at `M_max`. Torsion is already intrinsic and
integrates directly to
`W_t = c1 [omega atan(c2 omega) - ln(1 + c2^2 omega^2)/(2 c2)]`, with
`GJ_0 = c1 c2`.

Recasting also removes a length inconsistency: `kite_fem` infers `EI = P/(3v)`,
which is the true `EI` only for a one-metre element because `v` is already
normalised by element length. A moment-curvature law is length-independent.

Verified by pure end-moment solves, where the exact answer is a
constant-curvature arc at whatever curvature the fit prescribes -- exact at any
deflection, unlike the tip-load form, whose own inversion assumes linear
cantilever theory. Worst error 0.06% in bending, 1.5% in torsion
(`run_validation_benchmarks.py --case inflatable`, which completes
`kite_fem/examples/FEM_beam_verification.py`).

**Collapse is reported, not enforced.** A dropping post-collapse moment would
make the energy fall with curvature, i.e. an unbounded mechanism that
minimisation would simply run away from. `inflatable_beam_state` returns
`utilisation` (curvature over collapse curvature) and a `collapsed` flag, so
leaving the calibrated range is visible rather than silent. Axial and shear
stiffness are not covered by the fits and must be supplied from tube geometry.

## Element reference

| kernel | nodes | rot. DOF | key parameters | notes |
|--------|-------|----------|----------------|-------|
| `CableKernel` | 2 | no | `rest_length`, `stiffness` | `tension_only` cuts compression (C1); `slack_smoothing` rounds it |
| `PulleyKernel` | 3 | no | `rest_length`, `stiffness` | `rest_length` is the **whole rope**, both arms |
| `TimoshenkoBeamKernel` | 2 | both | `rest_length`, `ea/ga_2/ga_3/gj/ei_2/ei_3`, `gamma_0`, `omega_0` | 1-point integration; reference strains make curved members stress-free |
| `MembraneKernel` | 3 | no | `area`, `thickness`, `youngs_modulus`, `poisson_ratio`, `d0inv_*` | Pipkin relaxed energy; `slack_stiffness_ratio` is required, see below |

**Fabric needs a residual slack stiffness.** A fully slack region stores exactly
zero energy, so its Hessian block is exactly zero and IPOPT fails with
`Error_In_Step_Computation`. `MembraneKernel.slack_stiffness_ratio` (default
`1e-4`) blends a small fraction of the unrelaxed law back in:
`U = (1-r) U_relaxed + r U_taut`. The taut region is untouched because the two
coincide there. Do not set it to zero on a canopy that can go slack.

**Wrinkling is why energy minimisation is the right formulation here**, not a
workaround. The relaxed functional is the quasiconvex envelope, so the minimiser
lands on the wrinkled state directly instead of chasing the near-singular tangent
a residual-form Newton solve has to fight through. The demo shows the contrast: on
a panel clamped into a frame 6% smaller than itself the relaxed law converges in
31 iterations to a smooth billow, the unrelaxed one takes 211 and buckles into
mesh-scale crumple.

## Known traps, found the hard way

Every one of these cost real time to diagnose. They are properties of the
physics or the parameterisation, not bugs to be fixed.

- **Reference curvature is the quantity that breaks first on a real structure.**
  `omega_0 = psi / L0` with `psi` a Rodrigues vector grows like `tan(theta/2)`
  and is singular at `theta = pi`, so a short element bridging a large frame
  change is numerically hostile long before it is physically wrong. On a real
  LEI kite an independently-seeded roll per member put the worst element at 173
  degrees, `omega_0 = 753 /m` on a 17 mm element. The tube section is isotropic
  in roll, so `d1` is the only physically determined director and the roll is
  free: transporting it by minimal rotation over the beam network, and letting
  the LONG member own the joint, gives 10.6 /m.
- **The roll is a gauge only along a member, not node by node — and a
  mirror-symmetric structure needs mirror-consistent frames.** The element
  measures the relative rotation of its end frames, so rolls that differ between
  neighbours change the energy. Minimal-rotation transport over a network mixing
  a member that crosses the mirror plane (tangent maps as `-M t`) with members
  beside it (`+M t`) is not mirror-equivariant: on a real kite the far half
  arrived rolled by up to 60 degrees with positions symmetric to 4e-15 m, and
  the structure solved to one deterministic asymmetric shape.
  `symmetry.mirror_frames` fixes it with ONE director-sign matrix `S` for every
  node, `R_p = M R S`: each element's strains then map to a sign-flipped copy of
  its mirror element's, and every kernel here is even in each strain component.
  **Check any new kernel for that evenness before relying on it.**
- **An unbalanced free body is reported, not hidden.** A gravity-only load on a
  kite pinned at one bridle point slackens every line and the knots become a
  mechanism; the residual then sits at exactly their weight and no amount of
  solver tuning moves it. Codes that regularise with an absolute identity
  stiffness never show this, because it acts as a ground spring on every node.
  Read a stuck residual equal to some node's load as "this node has no load
  path", not as a conditioning problem.
- **Loads are dead (frozen) per solve.** That matches a staggered coupling loop,
  but a true follower pressure would not be conservative and could not be posed
  as a potential — worth remembering before anyone tries to fold aerodynamics
  into the same minimisation.
- **Applied moments need load stepping, and are exact only about a fixed axis.**
  The rotation DOF are Rodrigues vectors, whose chart is singular at a rotation
  of pi, so a node cannot be asked to turn most of the way round inside a single
  solve; ramp the moment and let the frame updates absorb it (four steps is
  enough for a full roll-up). Separately, a constant "dead" moment in 3-D is
  genuinely non-conservative and so has no potential at all: energy minimisation
  can only represent the fixed-axis case. Nodal *forces*, which is what an aero
  coupling applies, carry none of these caveats and are exact in one solve.
- **An element set with no elements is a degenerate `Function.map`.** Build the
  set only once it has something in it; `build_line_system` does.

## Conventions to preserve

- **Isolation.** NumPy and CasADi only.
- **Single source for every formula.** `rotations.py` holds the SO(3) maps and is
  written against an `xp` namespace so NumPy and CasADi share one implementation.
  `beam_strains` and `green_strain` are likewise called by both the kernels and
  the NumPy setup/diagnostic paths — do not restate a strain measure.
- **No CasADi across the module boundary.** Symbolics are created inside
  `energy.py` and stay there; `StructuralModel`, `StructuralState` and
  `StructuralSolution` are plain NumPy dataclasses.
- **Kernels are pure.** No element loops, no global indexing, no closing over
  per-element data; everything element-specific arrives through `params`.
- **`converged` is the physical verdict** (force balance within
  `force_tolerance`), with IPOPT's own verdict preserved as `ipopt_success` and
  `status`. IPOPT routinely reports `Search_Direction_Becomes_Too_Small` on stiff
  structures whose answer is already exact.
- **`move_limit` turns one `solve` into one trust-region step.** Default `None`
  keeps the unbounded behaviour. Set it (metres) when a slack-dominated model
  hands IPOPT a near-zero-curvature search direction: without bounds it answers
  with a step of order `1e4`, the objective overflows and the line search
  collapses into `Error_In_Step_Computation` before restoration can help. With
  it, drive an outer loop until the residual is met — and note that
  **`ipopt_success` no longer implies equilibrium**, because IPOPT reports
  success for the *boxed* problem while the iterate sits on the boundary;
  `converged` therefore keys off the force residual alone whenever a move limit
  is active. The same holds with `equalities`: IPOPT solves the constrained
  problem, whose optimum may be held by a reaction, and the reported residual
  INCLUDES that reaction -- zero only when the constrained state is an
  equilibrium of the free problem too, which makes it a check on the
  constraints. `validation/run_hanging_validation.py` is the worked example.
- **Tests assert against closed-form solutions and convergence orders, not
  against stored solver output.**
- **When you add, remove or rename a public function, dataclass, parameter or
  file, update this file and the docs in the same commit.**

## Open items

- The post-collapse branch of the inflatable tube is reported rather than
  modelled, and the fits cover only bending and torsion — axial and shear stay
  linear.
- Contact and self-contact are not modelled.
- `PotentialEnergy(parallelization=...)` exposes CasADi threaded maps; untested
  at scale, serial is the default.

## Physics references

- **Beam:** Simo & Vu-Quoc (1986) *Comput. Methods Appl. Mech. Engrg.* 58, 79–116.
- **Benchmarks:** Bathe & Bolourchi (1979) *Int. J. Numer. Meth. Engng.* 14,
  961-986; Bisshopp & Drucker (1945) *Q. Appl. Math.* 3, 272-275.
- **Rotation parameterisation:** Ibrahimbegović (1997) *Comput. Methods Appl.
  Mech. Engrg.* 149, 49–71.
- **Membrane wrinkling:** Pipkin (1986) *IMA J. Appl. Math.* 36, 85–99;
  Roddeman et al. (1987) *J. Appl. Mech.* 54, 884–892.
- **Aerostructural context:** Cayon, Gaunaa & Schmehl (2023) *Energies* 16, 3061.
