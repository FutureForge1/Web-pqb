# Web-PQB Benchmark Audit Checklist

This checklist is for manually curating `webpqb_benchmark.json` into a cleaner
`benchmark_v2` test set.

## Global Questions for Every Task

For each row in `webpqb_benchmark_audit.csv`, answer these first:

1. Is the task solvable from the provided initial state?
2. Is the evaluator reliable and aligned with the natural-language goal?
3. Is this task visually grounded, rather than mostly solvable from text-only cues?
4. Is the task category assignment actually correct?
5. Is this task a near-duplicate of another task already in the benchmark?
6. Is this task representative enough to keep in the final test set?

If any answer is clearly "no", mark the task as `drop` or `revise`.

## Category 1: Multi-path

The task should measure whether agents choose a direct, efficient route rather
than a longer but still successful route.

Audit questions:

1. Are there at least two natural ways to reach the goal?
2. Is one route clearly more direct or efficient than another?
3. Would a detour still plausibly succeed, making process quality observable?
4. Is the task visually grounded enough that screenshots matter?
5. Is the shortest path obvious only after reasoning over the page, not just from text pattern matching?

Drop or revise if:

- There is only one natural route.
- The alternative route is too artificial.
- Success almost fully determines process quality.

## Category 2: High-distraction

The task should contain visually plausible distractors or competing choices.

Audit questions:

1. Are there multiple candidate items/pages that look plausible?
2. Do the constraints matter on the page itself, not just in the goal text?
3. Would a shallow agent likely click the wrong thing at least once?
4. Is the distractor load visible in the interface?
5. Does the task require sustained filtering or comparison?

Drop or revise if:

- The page is actually simple despite a long goal.
- The distractors are weak or irrelevant.
- The task is hard only because the instruction is verbose.

## Category 3: Recovery

The task should force state recognition and recovery from a wrong start page.

Audit questions:

1. Is the injected `wrong_start_url` clearly off-task?
2. Is it still recoverable without being trivial?
3. Does the wrong page create believable confusion?
4. Does the task require recognizing the current state before acting?
5. Is recovery itself an important part of the total trajectory quality?

Suggested severity labels:

- `low`: one-step recovery, obvious wrong page
- `medium`: requires re-orientation and at least one navigation decision
- `high`: wrong page contains plausible distractors or misleading affordances

Drop or revise if:

- The wrong page is too easy to escape.
- The wrong page accidentally leaks the answer.
- The injected state makes the task effectively unsolvable.

## Audit Workflow

1. Start with a balanced sample from each category.
2. Mark each task as `keep`, `drop`, or `revise`.
3. Record short notes for every dropped or revised task.
4. After one pass, rebalance by site and difficulty.
5. Freeze a `benchmark_v2` split only after manual review.

## Recommended Outputs

- `webpqb_benchmark_audit.csv`: working spreadsheet for review
- `webpqb_benchmark_v2.json`: final curated benchmark
- `benchmark_card.md`: benchmark description for the paper / repo
