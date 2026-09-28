# ARC-X: Recursive World-Model Reasoning Beyond HRM and TRM

## Research thesis and engineering specification

**Version:** 0.1 research draft  
**Repository:** `orbitronsystems/arcx-thesis`  
**Status:** Pre-experimental; all ARC-X performance claims are hypotheses until measured.

---

## Abstract

ARC-X is a proposed agent architecture for interactive abstract-reasoning environments. It combines deterministic multimodal perception, a structured canonical state, an executable world model, explicit goal inference, information-directed exploration, model-based planning, prediction verification, counterexample-driven repair, persistent memory, and a compact recursively improving reasoning network.

The design is motivated by the reported strengths of Hierarchical Reasoning Models (HRM) and Tiny Recursive Models (TRM): deep supervision, repeated latent refinement, parameter efficiency, and test-time computation can produce strong generalization from small datasets. ARC-X extends those ideas from static answer prediction to interactive scientific discovery. Instead of recursively refining only an answer grid, ARC-X recursively refines a **belief state** containing an environment theory, goal hypotheses, predicted consequences, and a plan. Real actions are reserved for experiments or verified plan prefixes, and every executed action produces evidence for model revision.

The central research question is:

> Can recursive latent refinement become more general and reliable when it operates over explicit, verifiable world-model state rather than over an answer representation alone?

This document defines the architecture, algorithms, experimental protocol, ablations, limitations, and thesis structure required to answer that question without conflating engineering targets with empirical results.

---

## 1. Contributions and Claims

ARC-X should be presented as five testable contributions, not as a claim that recursion alone creates general intelligence.

### C1 — Recursive belief refinement
A small tied network repeatedly improves a structured belief state containing mechanics, goals, uncertainty, and candidate plans.

### C2 — Verification-grounded recursion
The recursive controller receives exact transition deltas and prediction errors, allowing each reasoning iteration to correct a falsifiable theory.

### C3 — Separation of dynamics and objectives
Mechanics learning and goal inference are distinct latent variables and distinct evaluation targets.

### C4 — Action-efficient test-time discovery
The agent chooses between internal simulation, information-seeking experiments, and goal-directed actions under an explicit action budget.

### C5 — Local repair with regression protection
A failed prediction becomes a typed counterexample. A proposed repair is accepted only if it explains the new evidence without breaking previously verified transitions.

These claims must be validated against baselines and ablations. They must not be stated as established results before experiments are complete.

---

## 2. Relation to HRM and TRM

HRM and TRM address supervised input-to-output puzzles. ARC-X addresses partially observed interactive environments. This distinction determines the design.

| Dimension | HRM | TRM | ARC-X proposal |
|---|---|---|---|
| Primary task | Static puzzle answer | Static puzzle answer | Interactive environment control |
| Recurrent state | Two latent features at different update frequencies | Current answer `y` plus latent reasoning `z` | Belief state: dynamics, goals, graph, uncertainty, memory |
| Supervision | Deep supervision over answer prediction | Deep supervision over iterative answer improvement | Mixed: transition evidence, goal evidence, verification, reward/completion |
| Recursion target | Latent representation and output | Latent reasoning and output | Executable theory and action policy |
| Environment interaction | None | None | Required; actions alter future observations |
| Error signal | Answer loss | Answer and halt losses | Prediction mismatch, goal contradiction, failed plan, unsafe action |
| Verification | Label comparison | Label comparison | Predicted state versus observed state plus replay |
| Test-time adaptation | Iterative inference | Iterative inference | Belief update, model revision, goal revision, skill verification |
| Main risk | Fixed-point/gradient assumptions | Overfitting, compute cost, deterministic output | Model misspecification and costly exploration |
| Core advantage | Efficient effective depth | Simple tiny recursive refinement | Recursive reasoning grounded in falsifiable interaction |

ARC-X should adopt from TRM the practical insight that a single tied network with a separate answer state and reasoning state may be preferable to a biologically motivated hierarchy. It should not assume that the same architecture is automatically optimal for all grid sizes or modalities. For large visual contexts, spatial encoders and attention may be necessary; for compact symbolic contexts, MLP-style mixing may be competitive.

---

## 3. ARC-X Research Hypothesis

Let an environment produce observations `o_t` after actions `a_t`. The agent maintains a belief:

```text
b_t = (s_t, M_t, G_t, Γ_t, U_t, m_t)
```

where:

- `s_t` is the canonical state;
- `M_t` is a set of executable dynamics hypotheses;
- `G_t` is a set of goal hypotheses;
- `Γ_t` is the verified and hypothesized state graph;
- `U_t` is uncertainty and risk information;
- `m_t` is episodic, semantic, and procedural memory.

The recursive reasoner applies a tied update operator:

```text
b_(k+1) = R_θ(x_t, b_k, e_t)
```

where `x_t` is the current structured observation and `e_t` is recent evidence. The operator should improve one or more of:

1. transition prediction accuracy;
2. goal posterior calibration;
3. plan validity and efficiency;
4. action safety;
5. expected information gain.

The core hypothesis is:

> Recursive refinement over a verifiable belief state should generalize better to unseen mechanics and objectives than recursive answer refinement without an explicit environment model, at comparable inference compute.

---

## 4. System Architecture

```text
Raw grid/screen
      │
      ▼
Deterministic + learned perception
      │
      ▼
Canonical state and delta
      │
      ├──────────────► State graph and replay store
      ├──────────────► Memory and skill retrieval
      └──────────────► Recursive belief reasoner
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
   Mechanics update      Goal update        Experiment/planning
          │                   │                   │
          └───────────────────┼───────────────────┘
                              ▼
                       Internal simulator
                              │
                       Verification gate
                              │
                   ┌──────────┴──────────┐
                   ▼                     ▼
             Execute action        Reject/reason/replan
                   │
                   ▼
             New observation
                   │
             Prediction comparison
                   │
          Match ───┴─── Mismatch
           │                 │
       Commit evidence   Counterexample → local repair
```

### Design rule

The recursive network proposes and ranks hypotheses. Deterministic components remain authoritative for parsing, hashing, graph operations, replay, and comparison of observed transitions.

---

## 5. Recursive ARC-X Reasoner

### 5.1 State variables

Use two principal recurrent states, inspired by the useful reinterpretation of TRM:

- `y`: current structured proposal — candidate plan, predicted outcome, and goal-relevant state;
- `z`: latent reasoning state — compressed evidence, unresolved alternatives, causal explanations, and repair context.

Unlike TRM's fixed answer grid, `y` is a structured multi-head proposal. Unlike HRM's assumed hierarchy, the two states are functional roles rather than biological levels.

```python
@dataclass
class BeliefProposal:
    mechanics: list[MechanicsHypothesis]
    goals: list[GoalHypothesis]
    plan_candidates: list[Plan]
    predicted_outcomes: list[PredictedTransition]
    action_scores: dict[Action, float]
    confidence: float

@dataclass
class RecursiveContext:
    canonical_state: CanonicalState
    recent_deltas: list[TransitionDelta]
    counterexamples: list[Counterexample]
    graph_summary: GraphSummary
    memory_hits: list[MemoryItem]
    budget: ComputeBudget
```

### 5.2 Tied update operator

```python
def latent_recursion(context, proposal, z, steps):
    for _ in range(steps):
        z = reasoner(context, proposal, z)
        proposal = proposal_head(context, proposal, z)
    return proposal, z


def deep_recursion(context, proposal, z, inner_steps=6, outer_steps=3):
    with torch.no_grad():
        for _ in range(outer_steps - 1):
            proposal, z = latent_recursion(
                context, proposal, z, inner_steps
            )

    proposal, z = latent_recursion(context, proposal, z, inner_steps)
    return proposal.detach(), z.detach()
```

The production implementation should support a supervised mode for synthetic environments and a test-time adaptation mode for interactive levels. Full backpropagation through every recursion is not mandatory for all variants; it must be an experimental factor rather than an assumption.

### 5.3 Training losses

A composite objective may be used:

```text
L = λ_transition L_transition
  + λ_goal L_goal
  + λ_plan L_plan
  + λ_action L_action
  + λ_verify L_prediction_error
  + λ_halt L_halt
  + λ_complexity L_model_complexity
  + λ_calibration L_uncertainty
```

The weights must be reported and tuned only on development environments. Test environments must remain held out.

### 5.4 Adaptive computation

ARC-X should use halting as a policy over **reasoning iterations**, not as an unsupported confidence shortcut. Halt only when:

- the best proposal is stable across iterations;
- predicted transition confidence exceeds a threshold;
- no unresolved contradiction is active;
- the execution gate approves the candidate action;
- an additional iteration has lower expected value than its compute cost.

The system should log both the number of recursive iterations and the reason for halting.

---

## 6. Canonical State and Evidence

The state encoder must preserve exact information needed for replay while providing compact features to the reasoner.

```python
@dataclass(frozen=True)
class CanonicalState:
    level_id: str
    frame_id: int
    grid: tuple[tuple[int, ...], ...]
    entities: tuple[Entity, ...]
    relations: tuple[Relation, ...]
    topology: Topology
    player: PlayerState | None
    resources: tuple[tuple[str, float], ...]
    environment: tuple[tuple[str, str], ...]
    state_hash: str
```

The state hash must be invariant to timestamps, model versions, and confidence metadata. Confidence is stored separately.

A delta is evidence, not merely a visual difference:

```text
PLAYER_MOVED
OBJECT_MOVED
OBJECT_CREATED
OBJECT_REMOVED
RESOURCE_CHANGED
DOOR_OPENED
DOOR_CLOSED
HAZARD_TRIGGERED
TERMINAL_STATE_REACHED
```

The learned reasoner receives deltas; the verifier compares canonical states and typed differences.

---

## 7. World Model and Counterexample Repair

### 7.1 Hypothesis representation

Each mechanics hypothesis contains:

```text
preconditions
transition rule
affected entities
resource effects
temporal assumptions
complexity
supporting evidence
contradictions
confidence
```

The executable model is a weighted set of candidate programs, not an opaque prediction vector alone.

### 7.2 Verification loop

```python
def execute_verified(action, belief):
    prediction = belief.world_model.predict(belief.state, action)

    if not safety_gate.approve(prediction, belief):
        return Decision.REPLAN

    observation = environment.step(action)
    actual = perception.parse(observation)
    result = verifier.compare(prediction, actual)

    recorder.append(belief.state, action, prediction, actual, result)

    if result.matches:
        belief.commit_verified_transition(action, actual)
        return Decision.CONTINUE

    counterexample = Counterexample.from_result(
        state=belief.state,
        action=action,
        prediction=prediction,
        actual=actual,
        result=result,
    )
    belief.add_counterexample(counterexample)
    belief.repair_locally(counterexample)
    return Decision.RECOVER
```

### 7.3 Repair acceptance

A repair is accepted only if:

1. it explains the triggering counterexample;
2. it does not break all previously verified transitions;
3. it does not increase model complexity without evidence;
4. its uncertainty is propagated to planning;
5. it is versioned and replayable.

This is the main distinction between recursive refinement of an answer and recursive refinement of an executable theory.

---

## 8. Goal Inference

Mechanics and goals must not be collapsed into one prediction head. A system can correctly learn that a door opens when a switch is pressed while still being wrong about whether the door is the terminal objective.

Each goal hypothesis stores:

```text
goal_type
parameters
prior
evidence
contradictions
reachability
expected completion signature
confidence
```

Goal updates combine structural evidence, observed environment behavior, and inverse planning. A candidate goal is not accepted merely because it is visually salient or reachable.

A goal experiment should be selected when its expected information gain exceeds its risk and action cost:

```text
EIG(a) = H(G | D) - E_o[H(G | D, a, o)]
```

where `H` is entropy over goal hypotheses.

---

## 9. Planning and Exploration

The action selector ranks actions using:

```text
Score(a) = α progress(a)
         + β information_gain(a)
         + γ novelty(a)
         - δ risk(a)
         - ε action_cost(a)
         - ζ uncertainty(a)
```

Planning operates on the state graph and simulator. It distinguishes:

- verified edges;
- model-predicted but unverified edges;
- failed edges;
- unexplored edges.

The planner should prefer a slightly longer verified path over a shorter high-risk path when RHAE and failure cost make that choice favorable.

---

## 10. Training and Evaluation

### 10.1 Training curriculum

```text
C0  deterministic movement
C1  one interaction mechanic
C2  multiple independent mechanics
C3  interacting mechanics
C4  hidden delayed effects
C5  ambiguous goals
C6  hazards and irreversible actions
C7  long-horizon compositions
C8  held-out mechanic combinations and visual variants
```

Training environments must be procedurally generated with held-out rule compositions. Public benchmark levels must not be used to tune the architecture repeatedly without documenting the exposure.

### 10.2 Baselines

- direct multimodal model with action prompting;
- hierarchical recurrent baseline;
- TRM-style recursive answer/state baseline;
- symbolic planner without learned reasoning;
- learned dynamics policy without verification;
- ARC-X without recursive reasoner;
- full ARC-X.

### 10.3 Primary metrics

- completion rate;
- RHAE or the benchmark's official efficiency metric;
- real environment actions;
- valid-action rate;
- transition prediction accuracy;
- unseen-transition accuracy;
- goal identification accuracy before terminal reward;
- recovery success after mismatch;
- calibration of confidence;
- compute per solved level;
- recursive iterations per decision.

### 10.4 Statistical protocol

Report per-level results, not only aggregate averages. Use fixed seeds, confidence intervals or bootstrap intervals, and paired comparisons on identical levels. Separate development, validation, and final test environments. Report failures by taxonomy rather than removing them as outliers.

---

## 11. Ablation Matrix

| Variant | Removed component | Main question |
|---|---|---|
| A0 | Direct action baseline | How hard is the task without structure? |
| A1 | Canonical state | Does exact representation matter? |
| A2 | Recursive refinement | Does repeated belief improvement help? |
| A3 | World model | Does theory-building improve efficiency? |
| A4 | Verification | Are predictions useful without reality checks? |
| A5 | Counterexample repair | Does local repair improve recovery? |
| A6 | Goal separation | Does explicit objective inference matter? |
| A7 | State graph | Does systematic exploration beat local policy? |
| A8 | Memory transfer | Does verified procedural knowledge transfer? |
| A9 | Adaptive halting | Is dynamic compute better than fixed recursion? |
| A10 | Full ARC-X | Combined system result |

The primary comparison should be between A10 and each one-component removal, with identical compute and action budgets wherever possible.

---

## 12. Limitations and Falsification Criteria

ARC-X should be considered unsuccessful or incomplete if:

- the state parser introduces more errors than it removes;
- world-model planning increases actions despite higher compute;
- recursive reasoning does not improve held-out mechanics;
- repair fixes local examples but causes replay regressions;
- explicit goals do not improve ambiguous environments;
- performance depends on memorized public-level layouts;
- the architecture cannot run within competition constraints.

A negative result is scientifically valuable if the ablations identify which assumption failed.

---

## 13. Thesis Structure

1. **Introduction** — Interactive generalization as theory construction.
2. **Background** — ARC, interactive agents, HRM, TRM, world models, planning.
3. **Problem formulation** — Unknown dynamics, unknown goals, action budget.
4. **ARC-X architecture** — System overview and interfaces.
5. **Recursive belief refinement** — `y/z` state design, deep supervision, halting.
6. **Perception and canonical state** — Exact representation and deltas.
7. **Executable world model** — Rules, simulation, versioning.
8. **Goal inference and exploration** — Hypotheses, information gain.
9. **Verification and repair** — Counterexamples and regression replay.
10. **Memory and skill transfer** — Evidence-gated reuse.
11. **Experimental protocol** — Data, baselines, metrics, statistical design.
12. **Results** — Completion, efficiency, prediction, goal accuracy.
13. **Ablations and failure analysis** — Causal contribution of components.
14. **Discussion** — What recursion adds beyond hierarchy.
15. **Limitations and ethics** — Scope, reproducibility, claims.
16. **Conclusion** — Verified recursive reasoning as a research direction.

---

## 14. Final Position

ARC-X should not claim to replace TRM with a larger or more complicated network. The stronger contribution is to place a compact recursive learner inside a verifiable agent loop:

```text
recursive latent improvement
+ explicit state
+ executable dynamics
+ explicit goals
+ internal simulation
+ real-world verification
+ counterexample repair
```

TRM provides a strong design lesson: repeated refinement with a small tied model can outperform naïvely increasing capacity. ARC-X extends that lesson to interactive environments by making the object of refinement an executable belief and by grounding every important update in evidence.

The resulting research program is more ambitious than HRM or TRM, but it is also more falsifiable: every component has an interface, a metric, an ablation, and a failure condition.
