# ARC-X: Recursive Verified World-Model Reasoning for Novel Interactive Environments

**Milestone 2 Submission — Research Paper**  
**Project:** ARC-X (Adaptive Reasoning & World-Model Architecture)  
**Author/Team:** Orbitron Systems  
**Submission date:** 30 September 2026  
**Code and supplementary material:** https://github.com/orbitronsystems/arcx-thesis

> **Evidence policy.** This paper distinguishes reported results from proposed methods and engineering targets. Unless explicitly labelled as a result, a number is not claimed as an ARC-X measurement. ARC-X has not been granted performance by association with prior systems.

---

## Abstract

Interactive abstract reasoning requires more than predicting an action from an image. An agent must infer the latent mechanics of an unfamiliar environment, determine what objective is being pursued, explore under an action budget, and recover when its theory is wrong. We present **ARC-X**, a hybrid architecture for this setting that combines deterministic multimodal state extraction, recursive latent refinement, executable world models, explicit goal hypotheses, information-directed exploration, model-based planning, prediction verification, counterexample-driven repair, and evidence-gated skill transfer.

ARC-X is motivated by recent results showing that small recursively applied networks can generalize strongly on static reasoning tasks. In particular, the Hierarchical Reasoning Model (HRM) and Tiny Recursive Model (TRM) demonstrate the value of deep supervision, repeated refinement, and parameter-efficient test-time computation. ARC-X preserves the useful computational idea—reusing a compact reasoner across multiple refinement steps—but changes the object of reasoning from a static answer grid to an interactive **belief state** containing dynamics hypotheses, goal hypotheses, uncertainty, graph structure, and candidate plans.

The central hypothesis is that recursive reasoning becomes more reliable and transferable when every important prediction is connected to an executable model and checked against subsequent observations. A real action is therefore treated as an experiment or a verified plan prefix. A mismatch becomes a typed counterexample; repairs are accepted only after regression replay over previously verified transitions. This design separates perception, dynamics learning, goal inference, planning, and control while allowing a learned reasoner to coordinate them.

We specify the architecture, formal problem, recursive update mechanism, verification protocol, evaluation methodology, ablations, limitations, and reproducibility requirements. The paper contributes a testable research program rather than unsupported claims of human-level performance.

**Keywords:** interactive reasoning, ARC-AGI, recursive learning, world models, model-based planning, goal inference, counterexample repair, test-time adaptation, multimodal agents.

---

## 1. Introduction

A static visual reasoning task asks an agent to transform an input into an output. An interactive abstract environment asks a harder question: **what should the agent do when it does not yet know what the objects, actions, rules, or objective mean?** The distinction is important. A system may recognize a visual pattern while still failing to understand which actions change the world, which changes are reversible, and which state is terminal.

ARC-AGI was introduced as a measure of abstraction and reasoning rather than memorization [1]. Later ARC work and interactive environments have continued to expose a gap between performance on familiar transformations and robust generalization to novel tasks [2]. The challenge is not solved by increasing the size of an action-prediction model alone. An agent must construct a compact theory of its current environment and use that theory to decide which experiment or action is justified.

Recent recursive approaches provide an important architectural clue. HRM uses two recurrent networks, deep supervision, and repeated computation to obtain large effective depth with a comparatively small parameter budget [3]. TRM simplifies this design to a single small network that recursively improves a candidate answer and a latent reasoning state, reporting strong results on several static puzzle benchmarks [4]. Independent analysis has also argued that deep supervision is a major contributor to HRM's performance, making the training objective and iterative refinement especially important [5].

However, a static answer-refinement model does not by itself solve interactive discovery. It does not necessarily represent verified action effects, maintain competing goals, distinguish an exploratory action from a goal-directed action, or repair an executable theory after an unexpected transition. ARC-X addresses this gap.

### 1.1 Research question

> Does recursive refinement over an explicit, verifiable belief state improve generalization and action efficiency in unfamiliar interactive environments compared with one-pass, hierarchical, recursive, and unverified agents?

### 1.2 Contributions

This paper makes the following contributions:

1. **Recursive belief refinement.** We define a compact tied reasoner whose recurrent state represents a candidate environment theory, objective hypotheses, uncertainty, and plans rather than only an answer grid.
2. **Verification-grounded interaction.** We place an executable world model and a deterministic comparison layer between proposed actions and environment execution.
3. **Separate mechanics and goal inference.** Dynamics hypotheses and objective hypotheses are represented and evaluated independently.
4. **Counterexample-driven repair.** Prediction failures are converted into typed evidence and used for local, versioned repair with historical replay checks.
5. **A reproducible evaluation protocol.** We specify baselines, metrics, held-out procedural environments, ablations, and reporting requirements that distinguish completion from action efficiency.

The target of 100% completion and 100 RHAE is an engineering objective, not a result established by this paper.

---

## 2. Background and Related Work

### 2.1 ARC-style abstraction and generalization

ARC frames intelligence as the ability to infer compact transformations from a small number of examples [1]. ARC-AGI-2 extends the challenge and documents continued difficulty for frontier systems [2]. Interactive ARC-AGI-3-style environments add temporal interaction, hidden mechanics, and action consequences. Static accuracy and interactive completion are therefore different metrics and must not be compared as if they were interchangeable.

### 2.2 Recursive reasoning with small networks

HRM proposes recursive hierarchical reasoning with two networks operating at different update frequencies and deep supervision across refinement steps [3]. TRM simplifies the approach by carrying a candidate solution and a latent reasoning representation through repeated updates using one small network [4]. Reported experiments in TRM show that a smaller recursively applied network can outperform larger alternatives on the authors' selected static benchmarks [4]. These results motivate ARC-X's use of tied recursive computation, but they do not establish that a direct TRM port solves interactive environments.

ARC-X adopts two practical lessons:

- effective reasoning depth can come from repeated application rather than only parameter scaling;
- an explicit candidate state and latent reasoning state can be more useful than an unnecessarily complex hierarchy.

ARC-X changes the target of refinement. Its candidate state is a structured proposal about the environment and its latent state stores unresolved evidence, causal explanations, and repair context.

### 2.3 Program synthesis and neuro-symbolic reasoning

High-performing ARC systems have frequently combined learned proposal mechanisms with explicit search, program induction, or test-time adaptation [6]. The common principle is compositionality: a model proposes structured hypotheses while a symbolic or algorithmic process checks whether they explain the evidence. ARC-X applies the same principle to interactive transitions rather than only static grid transformations.

### 2.4 Interactive agents and executable world models

Interactive agents benefit from memory, tools, feedback, recovery, and long-horizon supervision. The Tufa Labs Duck Harness is an example of an open-source ARC-AGI-3 inference harness that emphasizes orchestration and tool use [7]. ARC-X is complementary: it focuses on a formalized belief state, executable transition hypotheses, goal inference, and verification. Any direct performance comparison must use the same competition release, action budget, environment set, and evaluation metric.

### 2.5 Positioning

| Capability | Static recursive solver | Direct multimodal agent | ARC-X |
|---|---:|---:|---:|
| Repeated latent refinement | Yes | Optional | Yes |
| Deterministic canonical state | Usually no | Usually no | Yes |
| Executable dynamics model | No | Usually no | Yes |
| Explicit goal hypotheses | Usually implicit | Implicit or prompt-based | Yes |
| State/action graph | No | Optional | Yes |
| Prediction verification | Label-based | Inconsistent | Required |
| Counterexample repair | Not central | Prompt-dependent | Versioned and replay-checked |
| Interactive exploration | No | Yes, often unguided | Information-directed |
| Evidence-gated skill transfer | Limited | Variable | Required |

---

## 3. Problem Formulation

Let an environment have latent state `s_t`, observation `o_t`, action `a_t`, transition dynamics `T`, and terminal objective `G`. The agent receives observations but not the rules or objective:

```text
s_(t+1) ~ T(s_t, a_t)
 o_t = O(s_t)
```

At time `t`, ARC-X maintains a belief state:

```text
b_t = (ŝ_t, M_t, G_t, Γ_t, U_t, m_t)
```

where:

- `ŝ_t`: canonical structured state;
- `M_t`: competing executable mechanics models;
- `G_t`: goal hypotheses;
- `Γ_t`: state graph with verified and hypothesized edges;
- `U_t`: uncertainty, risk, and confidence;
- `m_t`: working, episodic, semantic, and procedural memory.

The agent must choose an action that balances progress, information, cost, and risk. The ideal action is not necessarily the action with the highest immediate reward; it may be an experiment that distinguishes two plausible theories while preserving recoverability.

A general action utility is:

```text
U(a) = α Progress(a)
     + β InformationGain(a)
     + γ Novelty(a)
     - δ Risk(a)
     - ε ActionCost(a)
     - ζ Uncertainty(a)
```

The objective is to maximize completion and action efficiency subject to a limited number of real environment actions.

---

## 4. ARC-X Architecture

```text
Raw grid or screen
        │
        ▼
Perception and object extraction
        │
        ▼
Canonical state + transition delta
        │
   ┌────┼─────────┬───────────┐
   ▼    ▼         ▼           ▼
Graph Memory  World model  Goal registry
   └────┬─────────┴───────────┘
        ▼
Recursive belief reasoner
        │
 ┌──────┼─────────┬─────────┐
 ▼      ▼         ▼         ▼
Explore Plan   Repair   Reflect
        │
        ▼
Internal simulation
        │
        ▼
Safety and execution gate
        │
        ▼
Real action → new observation
        │
        ▼
Prediction comparison → commit or repair
```

The architecture follows a strict separation of responsibilities:

- **Deterministic computation** handles parsing, hashing, graph operations, state comparison, and replay.
- **Learned reasoning** handles ambiguity, hypothesis generation, goal inference, causal explanations, and difficult strategy selection.
- **The execution gate** prevents unverified neural outputs from becoming environment actions.

---

## 5. Canonical Multimodal State

A raw grid or image is converted into a stable, serializable state:

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

The perception pipeline begins with deterministic operations:

1. validate dimensions and palette;
2. segment colors or visual regions;
3. extract connected components;
4. compute bounding boxes, centroids, shape descriptors, and orientation;
5. compute relations such as adjacency, containment, alignment, and overlap;
6. infer walkable and blocked topology;
7. track object identities across frames;
8. compare consecutive states to produce typed deltas.

The state hash excludes timestamps, confidence metadata, and model versions. This enables caching, cycle detection, graph construction, and exact replay.

A delta is represented as evidence:

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

When visual ambiguity remains, multiple canonical interpretations may be retained with confidence rather than forcing a single unsupported answer.

---

## 6. Recursive Belief Refinement

### 6.1 Two functional recurrent states

ARC-X uses two principal carried states, inspired by the useful interpretation of TRM but not by a claim of biological hierarchy:

- `y`: a structured proposal containing candidate mechanics, goals, predicted outcomes, and plans;
- `z`: a latent reasoning state containing compressed evidence, unresolved alternatives, causal explanations, and repair context.

```python
@dataclass
class BeliefProposal:
    mechanics: list[MechanicsHypothesis]
    goals: list[GoalHypothesis]
    plans: list[Plan]
    predictions: list[PredictedTransition]
    action_scores: dict[Action, float]
    confidence: float
```

### 6.2 Tied recursive update

```python
def latent_recursion(context, proposal, z, steps=6):
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

The tied reasoner is not allowed to be the sole authority for observed transitions. It proposes models and plans; deterministic verification decides whether an observed transition matches a prediction.

### 6.3 Training objective

For synthetic environments with known transition functions and goals, ARC-X can use:

```text
L = λ_dynamics  L_transition
  + λ_goal      L_goal
  + λ_plan      L_plan
  + λ_action    L_action
  + λ_verify    L_prediction_error
  + λ_halt      L_halt
  + ��_complex   L_model_complexity
  + λ_calib     L_uncertainty_calibration
```

Training should include correct trajectories, near-miss trajectories, and counterexamples of the form:

```text
(state, action, predicted_state, observed_state, difference)
```

### 6.4 Adaptive computation

The agent may halt recursive reasoning only when:

- proposal rankings are stable;
- active contradictions have been resolved or explicitly budgeted;
- the predicted outcome passes safety checks;
- the best action is aligned with the current goal posterior;
- another recursion has lower expected value than its compute cost.

The system records the number of recursion steps, confidence, and halt reason for every decision.

---

## 7. Executable World Model

The world model contains explicit, testable rules:

```text
preconditions
transition effects
resource effects
temporal assumptions
risk annotations
supporting evidence
contradictions
confidence
```

It exposes:

```python
predict(state, action) -> PredictedState
step(state, action) -> PredictedState
replay(trajectory) -> ReplayReport
goal_reached(state) -> bool
repair(counterexample) -> ModelVersion
confidence() -> float
```

A model version records its parent, evidence, rule changes, tests, and prediction accuracy. Silent mutation is prohibited.

### 7.1 Model selection

Among models that explain the observed evidence, prefer the simplest executable model:

```text
M* = argmin_M [Complexity(M) + λ · PredictionError(M)]
```

This prevents the reasoner from accumulating unsupported rules merely because they can explain one transition.

---

## 8. Verification and Counterexample Repair

Before acting, ARC-X performs the following loop:

```python
def execute_verified(action, belief):
    prediction = belief.world_model.predict(belief.state, action)

    if not safety_gate.approve(prediction, belief):
        return REPLAN

    observation = environment.step(action)
    actual = perception.parse(observation)
    result = verifier.compare(prediction, actual)
    recorder.append(belief.state, action, prediction, actual, result)

    if result.matches:
        belief.commit_verified_transition(action, actual)
        return CONTINUE

    counterexample = Counterexample(
        state=belief.state,
        action=action,
        predicted_state=prediction,
        observed_state=actual,
        difference=result.differences,
    )
    belief.add_counterexample(counterexample)
    belief.repair_locally(counterexample)
    return RECOVER
```

A repair is accepted only if it:

1. explains the triggering mismatch;
2. passes replay against historical verified transitions;
3. does not introduce unsupported complexity;
4. propagates residual uncertainty to planning;
5. receives a new version identifier.

Failure categories include perception, object identity, relation extraction, transition, interaction, physics, resource, goal, planning, execution, and memory-transfer errors.

This loop is the principal distinction between a model that repeatedly guesses and an agent that learns from interaction.

---

## 9. Goal Inference and Exploration

Mechanics learning and goal inference are separate because knowing how an object behaves does not establish why it exists.

Each goal hypothesis contains:

```text
goal_type
parameters
prior
evidence
contradictions
reachability
terminal signature
confidence
```

Candidate objectives may include reaching a region, collecting objects, opening a door, triggering a state, satisfying an ordered sequence, maximizing a resource, or surviving for a fixed duration.

The agent selects an experiment when it has high expected information gain and acceptable risk:

```text
EIG(a) = H(G | D) - E_o[H(G | D, a, o)]
```

The planner operates over a graph that distinguishes verified, predicted, failed, and unexplored edges. It should prefer an action that is slightly longer but verified and recoverable when the shorter alternative is uncertain or irreversible.

---

## 10. Memory and Skill Transfer

ARC-X uses four memory levels:

- **Working memory:** recent observations and transitions;
- **Episodic memory:** level-specific events, failures, and repairs;
- **Semantic memory:** generalized mechanics and relations;
- **Procedural memory:** verified reusable skills.

A skill is represented as:

```text
preconditions
procedure
expected effects
confidence
provenance
verification status
```

Transferred knowledge is always a hypothesis in a new environment. The execution gate requires local evidence before treating a skill as verified.

---

## 11. Experimental Protocol

### 11.1 Data partitions

1. **Training:** procedurally generated environments with known simulators.
2. **Validation:** new seeds and visual variants.
3. **Out-of-distribution validation:** held-out combinations of known mechanics.
4. **Final test:** unseen combinations, mechanics, layouts, and goals.
5. **Official evaluation:** reported separately with its exact release and metric.

### 11.2 Curriculum

```text
C0: movement and topology
C1: one interaction mechanic
C2: multiple independent mechanics
C3: interacting mechanics
C4: hidden delayed effects
C5: ambiguous and deceptive goals
C6: hazards and irreversible actions
C7: long-horizon compositions
C8: held-out mechanic combinations
```

### 11.3 Baselines

- direct multimodal model with structured action output;
- HRM-style recurrent baseline;
- TRM-style recursive proposal baseline;
- deterministic symbolic planner without learned reasoning;
- learned dynamics/value policy without verification;
- ARC-X with verification removed;
- full ARC-X.

Comparisons must match observation history, action budget, available training environments, and approximate inference compute.

### 11.4 Metrics

Primary metrics:

- completion rate;
- official RHAE or equivalent efficiency metric;
- real environment actions;
- valid-action rate.

Diagnostic metrics:

- transition prediction accuracy;
- unseen-transition accuracy;
- goal identification accuracy before terminal reward;
- recovery success after mismatch;
- number of global resets;
- calibration error;
- recursive iterations per decision;
- internal simulations per real action;
- wall-clock and memory use.

### 11.5 Reporting

Each run must record:

```text
git commit
configuration hash
random seed
environment release
model versions
parameter count
recursive steps
actions
completion
RHAE
predictions and observations
repairs
resets
runtime
```

Results should be reported per level as well as in aggregate, with confidence intervals or bootstrap intervals where the sample permits.

---

## 12. Ablations

| Ablation | Removed component | Question |
|---|---|---|
| A0 | Direct action baseline | How difficult is the task without structure? |
| A1 | Canonical state | Does exact representation improve reliability? |
| A2 | Recursive refinement | Does iterative belief improvement help? |
| A3 | World model | Does theory-building reduce actions? |
| A4 | Verification | What happens when predictions are not checked? |
| A5 | Counterexample repair | Does local repair improve recovery? |
| A6 | Goal separation | Does explicit objective inference help? |
| A7 | State graph | Does systematic exploration reduce repetition? |
| A8 | Memory transfer | Does verified procedural knowledge transfer? |
| A9 | Adaptive halting | Is dynamic compute preferable to fixed recursion? |
| A10 | Full ARC-X | Combined result |

The most important comparison is A10 against each single-component removal under the same environment and action budget.

---

## 13. Expected Results and Falsification

ARC-X is expected to improve prediction accuracy, recovery, and action efficiency relative to unverified baselines. These are hypotheses, not reported results.

The proposal is falsified or requires revision if any of the following occurs:

- canonical perception errors outweigh its benefits;
- recursion does not improve held-out mechanics at matched compute;
- verification adds overhead without reducing action failures;
- repair fixes a mismatch but produces replay regressions;
- goal separation does not help ambiguous objectives;
- memory transfer causes more negative transfer than positive transfer;
- results depend on memorized layouts or public-level leakage;
- the complete system cannot meet runtime constraints.

Negative results should be reported rather than removed.

---

## 14. Limitations

First, canonicalization is not equivalent to perfect perception. Visual ambiguity, object identity switches, and uncertain topology can still corrupt the belief state. Second, an executable world model may be wrong in systematic ways, and local repair cannot guarantee recovery from every failure. Third, explicit goal hypothesis spaces may omit the true objective. Fourth, synthetic environments can produce misleading confidence if their mechanics are less diverse than official environments. Fifth, recursive computation increases internal cost even when it reduces real actions. Finally, performance on a small number of games cannot establish general intelligence; it only provides evidence about the tested capabilities.

ARC-X therefore treats generalization as an empirical question requiring held-out mechanics, cross-environment transfer, failure analysis, and reproducible evaluation.

---

## 15. Conclusion

HRM and TRM show that repeated computation with small networks can be a powerful alternative to simply increasing model size. ARC-X extends this insight from static answer refinement to interactive theory refinement. Its recurrent state represents what the agent believes about the world, what objective it is pursuing, how confident it is, and what evidence contradicts its current model.

The resulting loop is:

```text
observe
→ canonicalize
→ hypothesize
→ recursively refine belief
→ simulate
→ verify
→ act
→ compare prediction with reality
→ repair locally
→ transfer only verified knowledge
```

The proposed contribution is not an unsupported claim that a larger neural network solves ARC-AGI. It is a testable architecture for making reasoning falsifiable, action-efficient, and recoverable. If the experiments support the hypotheses, ARC-X will provide evidence that recursive learning is more useful when grounded in an executable and verifiable model of interaction. If they do not, the ablations will identify which assumptions fail.

---

## References

[1] François Chollet. **On the Measure of Intelligence.** arXiv:1911.01547, 2019. https://arxiv.org/abs/1911.01547

[2] François Chollet, Mike Knoop, Grayson Kamradt, Brian Landers, and Ian Pinkard. **ARC-AGI-2: A New Challenge for Frontier AI Reasoning Systems.** arXiv:2505.11831, 2025. https://arxiv.org/abs/2505.11831

[3] G. Wang, J. Li, Y. Sun, X. Chen, C. Liu, Y. Wu, M. Lu, S. Song, and Y. A. Yadkori. **Hierarchical Reasoning Model.** arXiv:2506.21734, 2025. https://arxiv.org/abs/2506.21734

[4] Alexia Jolicoeur-Martineau. **Less is More: Recursive Reasoning with Tiny Networks.** 2025. See the supplied manuscript and associated project materials for the reported HRM/TRM experiments.

[5] ARC Prize Foundation. **The Hidden Drivers of HRM's Performance on ARC-AGI.** 2025. https://arcprize.org/blog/hrm-analysis

[6] ARC Prize Foundation. **ARC Prize 2024: Technical Report.** arXiv:2412.04604, 2024. https://arxiv.org/abs/2412.04604

[7] Tufa Labs. **Duck Harness: The Duck — ARC-AGI-3 Inference Harness.** Open-source repository and technical materials, accessed 2026. https://github.com/Tufalabs/duck-harness

[8] John Jumper et al. **Attention Is All You Need.** NeurIPS, 2017. https://arxiv.org/abs/1706.03762

[9] Shaojie Bai, J. Zico Kolter, and Vladlen Koltun. **Deep Equilibrium Models.** NeurIPS, 2019. https://arxiv.org/abs/1909.01377

[10] Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. **Scaling LLM Test-Time Compute Optimally Can Be More Effective Than Scaling Model Parameters.** arXiv:2408.03314, 2024. https://arxiv.org/abs/2408.03314

---

## Reproducibility and artifact statement

The ARC-X research materials, engineering specification, experimental protocol, and diagrams are maintained at:

https://github.com/orbitronsystems/arcx-thesis

The repository currently contains the architecture and protocol. Measured ARC-X results, if added later, must include the corresponding code commit, configuration, environment release, seeds, trajectories, and raw metrics. No unmeasured target in this paper should be interpreted as an achieved score.
