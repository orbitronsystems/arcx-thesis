# Professional Experimental Protocol

## Objective

Evaluate whether ARC-X's recursive, verified belief architecture improves interactive generalization and action efficiency over direct, hierarchical, recursive, symbolic, and learned baselines.

## Data partitions

1. **Training:** procedurally generated environments with known simulator state.
2. **Validation:** new seeds and visual variants using known primitive families.
3. **OOD validation:** held-out combinations of known primitives.
4. **Final test:** unseen mechanics, layouts, and goal compositions.
5. **Competition evaluation:** official environments, reported separately from synthetic results.

No final-test trajectory may be used to tune thresholds, architecture, or prompts.

## Environment generator requirements

Each generated level must record:

```text
seed
primitive mechanics
composition graph
goal type
initial state
legal actions
transition function
terminal condition
shortest verified solution
risk annotations
```

The generator must support movement, collection, switches, doors, hazards, resources, delayed effects, ordering constraints, and deceptive visual salience.

## Baseline definitions

### Direct multimodal baseline
Receives the observation history and emits a structured action. No explicit graph, simulator, or repair memory.

### HRM-style baseline
Two recurrent modules and deep supervision over a structured action/output target. No explicit world-model verification.

### TRM-style baseline
One tied recurrent module with proposal state and latent state. It may use the same perception encoder but does not maintain executable dynamics hypotheses.

### Symbolic baseline
Deterministic parser plus hand-specified search and no learned recursive reasoner.

### Learned dynamics baseline
Learned transition/value model plus policy; no explicit counterexample repair or goal registry.

### ARC-X
Canonical state, recursive belief refinement, world-model hypotheses, goal registry, graph, simulator, verification gate, repair, memory, and adaptive compute.

## Metrics

### Completion
```text
completed levels / evaluated levels
```

### Efficiency
Use the official RHAE definition when available. Additionally report:

```text
action efficiency = reference action count / real actions
```

### Transition accuracy
```text
correct predicted typed deltas / tested transitions
```

### Goal accuracy
Goal selected before terminal reward and confirmed by terminal semantics.

### Recovery success
Fraction of prediction mismatches after which the agent reaches completion without global reset.

### Recursive efficiency
Completion and prediction accuracy as a function of recursive iterations and compute time.

## Reporting standard

Every result table must include:

- number of levels;
- seeds;
- model parameters;
- recursive steps;
- wall-clock time;
- real actions;
- internal simulations;
- failures and resets;
- confidence interval;
- code commit and configuration hash.

## Stop conditions

Terminate an experiment if a bug invalidates the protocol, data leakage is detected, or the environment generator's ground truth is inconsistent. Record the incident; do not silently discard it.
