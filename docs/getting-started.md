# Getting started

## Install

```sh
pip install git+https://github.com/awegroup/Billow.git
```

The runtime dependencies are NumPy and CasADi. IPOPT ships inside the CasADi
wheel, so there is nothing else to build.

For the demonstration and validation scripts, which draw figures and solve
reference boundary-value problems:

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

## A first solve

Two cables from fixed anchors to one free node, loaded downward:

```python
import numpy as np
from billow import MinimumEnergySolver, StructuralModel
from billow.elements import build_cable_elements

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

The cables have no stiffness against the load until they stretch, so the node
sags until the vertical components of two tensions balance 500 N. Nothing had to
be told that; it is where the energy is least.

## Reading a solution

`StructuralSolution` carries the answer and the evidence for it.

| field | meaning |
|---|---|
| `state` | the equilibrium configuration: positions, and frames where there are rotational DOF |
| `converged` | the **physical** verdict: force balance within `force_tolerance` |
| `ipopt_success` / `status` | IPOPT's own verdict, preserved separately |
| `residual_forces` / `residual_norm` | largest out-of-balance force over the free nodes [N] |
| `internal_forces` / `internal_moments` | what the elements carry |
| `strain_energy` | the objective at the solution [J] |
| `iterations`, `frame_updates` | cost |

!!! warning "`converged` and `ipopt_success` are different questions"
    IPOPT routinely reports `Search_Direction_Becomes_Too_Small` on stiff
    structures whose answer is already exact, so its verdict is not the one to
    key off. `converged` is a force balance.

    The gap widens when a `move_limit` or `equalities` are active: IPOPT then
    reports success for the *boxed* or *constrained* problem while the iterate
    sits on the boundary. `converged` keys off the force residual alone
    whenever a move limit is active, for exactly that reason.

## Reading a stuck residual

A residual that will not move is usually telling you something about the model,
not the solver.

!!! tip "A residual equal to some node's load means that node has no load path"
    A gravity-only load case on a kite pinned at one bridle point slackens every
    line, and the knots become a mechanism. The residual then sits at exactly
    their weight and no amount of solver tuning moves it.

    Billow reports that faithfully. Codes that add an absolute identity
    stiffness to the tangent hide it, because that acts as a ground spring on
    every node. Read it as "this node has nothing holding it", not as a
    conditioning problem.

## The wireframe model

For a pure line system — a bridle, a net, anything with no bending — build a
[`LineSystem`](api/wireframe.md) instead of assembling element sets by hand. It
owns the state across a sequence of solves, which is what an actuated or
iteratively coupled run needs:

```python
import numpy as np
from billow import build_line_system

system = build_line_system(
    nodes=np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    connectivity=[[0, 2], [1, 2]],
    rest_lengths=[1.0, 1.0],
    stiffness=[1e4, 1e4],
    fixed_nodes=[0, 1],
)

forces = np.zeros((3, 3))
forces[2] = [0.0, 0.0, -500.0]
system.solve(forces)

system.set_rest_length(0, 0.8)   # actuate: shorten one line
system.solve(forces)             # continues from the previous answer

print(system.tensions())         # per line, slack lines read 0
```

Positions persist between solves, rest lengths and stiffnesses are addressable
by your own line index, and none of that recompiles the solver — every numeric
input is an NLP parameter, so one build serves a whole actuation or stiffness
ramp.

!!! note "A pulley's rest length is the whole rope"
    Both arm indices address the one three-node element and report the length of
    both arms together. Line-system file formats disagree about this — some
    store the rope total on each arm, others split it — so
    `build_line_system` takes `pulley_rest_lengths` explicitly whenever the
    source does not simply split it. State the convention; do not let it be
    inferred.

## Solver settings worth knowing

```python
MinimumEnergySolver(
    model,
    tolerance=1e-8,            # IPOPT tol
    force_tolerance=1e-6,      # the force balance `converged` keys off [N]
    move_limit=None,           # metres; turns one solve into one trust-region step
    max_iterations=1000,
)
```

`move_limit` is the one to reach for when a slack-dominated model hands IPOPT a
near-zero-curvature search direction. Unbounded, IPOPT answers with a step of
order `1e4`, the objective overflows and the line search collapses into
`Error_In_Step_Computation` before restoration can help. With a move limit, each
`solve` advances by at most that much and an outer loop drives the residual
down.
