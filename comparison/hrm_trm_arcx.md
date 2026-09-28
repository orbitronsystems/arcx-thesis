# HRM and TRM to ARC-X: Design Comparison

## Executive summary

HRM and TRM are supervised recursive solvers. Their central contribution is computational: a small network can be applied repeatedly, with deep supervision, to obtain effective depth without a proportionally large parameter count. ARC-X adopts this principle but changes the reasoning object from a static answer to an interactive belief state.

## What ARC-X adopts

1. A carried answer/proposal state and a carried latent reasoning state.
2. Tied recursive updates rather than a large stack of independent layers.
3. Deep supervision across refinement steps for synthetic training.
4. Adaptive computation with explicit halting diagnostics.
5. Small models with more test-time computation when the task requires it.
6. Careful ablations instead of biological narratives as the primary explanation.

## What ARC-X changes

1. **Target:** an executable environment theory, not only an output grid.
2. **Evidence:** transitions, deltas, counterexamples, and terminal signals.
3. **Objective:** jointly improve dynamics, goals, plans, and uncertainty.
4. **Action:** no action is executed solely because a neural head selected it.
5. **Recovery:** errors trigger typed repair and historical replay.
6. **Generalization:** held-out mechanics and goal compositions, not only transformed examples.

## Why a direct TRM port is insufficient

A direct TRM port can repeatedly refine an action sequence or predicted grid, but it lacks an explicit representation of:

- what action effects are known;
- which transitions are verified;
- which goal is currently believed;
- whether an action is exploratory or goal-directed;
- how a failed prediction changes the model;
- which knowledge transfers safely to a new level.

ARC-X therefore uses TRM-like recursion as the controller over structured state rather than as the complete agent.

## Proposed research comparisons

The fairest comparisons are compute-matched:

- same parameter budget;
- same number of recursive calls;
- same observation history;
- same action budget;
- same training environments;
- separate development and held-out test worlds.

The paper should compare both accuracy and efficiency. A model that solves more levels while using substantially more real actions may not be superior under interactive evaluation.

## Predicted outcomes, explicitly labeled as hypotheses

- Recursive belief refinement should improve unseen-mechanic prediction over one-pass reasoning.
- Verification should improve recovery and reduce catastrophic continuation after a wrong theory.
- Goal separation should help environments with attractive but non-terminal objects.
- A state graph should reduce repeated exploration.
- The full system should outperform a direct recursive policy under equal real-action budgets.

These are hypotheses, not results.
