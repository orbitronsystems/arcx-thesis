# Evidence Policy

## Evidence labels

Every literature statement and benchmark statement must receive one label:

- **Verified-source:** directly supported by an official paper, competition page, repository, or measured run.
- **Reported:** reproduced from a source but not independently replicated by ARC-X.
- **Hypothesis:** proposed explanation to be tested.
- **Engineering target:** desired performance, not an achieved result.
- **Unknown:** insufficient evidence.

## Rules

1. Never convert a target into a result.
2. Never compare scores across incompatible tasks without explaining the metric and dataset.
3. Separate static ARC accuracy from interactive ARC-AGI-3 completion and RHAE.
4. Cite the original paper or official result before secondary commentary.
5. Record access date and commit or version for code.
6. Do not claim that a method generalizes to unseen environments merely because it performs well on transformed examples.
7. Report negative results and failed ablations.
8. Reproduce third-party claims only after matching their task, split, preprocessing, and number of attempts.

## Required source registry fields

```text
source_id,title,authors,year,source_type,url,task,metric,reported_result,evidence_label,access_date,notes
```

## Thesis language guidance

Use:

> "The source reports ..."

for third-party results.

Use:

> "We hypothesize ..."

for ARC-X claims not yet tested.

Use:

> "Our experiment measured ..."

only after the run artifacts and configuration are available.
