# Web-PQB v2: Category 2 `High-Distraction` Construction Protocol

## 1. Final Decision

Category 2 should **not** be built by passively mining naturally cluttered VWA tasks.

Category 2 should be built as a **controlled task-construction benchmark**:

- start from an existing VWA base task
- keep the original task goal semantics
- inject one or more **strong near-miss traps**
- patch the evaluator so the trap is explicitly disallowed
- validate with rollout statistics that the modified task induces more distraction than the original base task

In short:

> Category 2 is a paired, controlled, distraction-augmented benchmark.


## 2. What Category 2 Is Supposed to Measure

`High-Distraction` is not "a visually busy page".

It is a task where:

- multiple candidates appear plausible
- one or more visually salient candidates are tempting but wrong
- the agent is likely to produce:
  - wrong clicks
  - near-miss selections
  - short detours
  - suboptimal path choices

The benchmark should test:

- resistance to visual temptation
- fine-grained constraint tracking
- discrimination among similar candidates
- robustness under misleading but solvable conditions


## 3. Construction Unit

Every Category 2 sample must be a **pair**:

- `base version`: original VWA task
- `distraction version`: modified version of the same task

This paired design is mandatory because later experiments should compare:

- same site
- same goal
- same evaluator target
- different distraction strength

Without pairing, it becomes much harder to argue that performance differences come from distraction rather than task mismatch.


## 4. Base Task Selection Rules

Only choose base tasks that satisfy most of the following:

- the goal is decided by selecting among multiple candidates
- the task has at least one strong visual or attribute-based constraint
- the correct answer is still uniquely identifiable
- the task naturally happens on:
  - shopping category/listing pages
  - shopping search result pages
  - classifieds listing/search pages
  - reddit search/result/post lists with similar threads
- the evaluator is patchable and can identify the correct target robustly

Avoid base tasks that are:

- single-hop tasks with almost no candidate competition
- brittle evaluator tasks
- tasks whose answer is already trivially exposed
- tasks that become unsolvable once any trap is added


## 5. Site Priority

Recommended priority order:

1. `shopping`
2. `classifieds`
3. `reddit` as a smaller supplement

Rationale:

- `shopping` is best for visual competition, ranking traps, and attribute confusion
- `classifieds` is strong for listing confusion and near-duplicate entries
- `reddit` can support fine-grained thread confusion, but is less ideal for dense visual candidate competition


## 6. Final Construction Templates

Category 2 should use the following three templates.

### Template A: Negative Constraint Trap

Definition:

- preserve the original goal
- add an explicit negative constraint that excludes the most visually attractive or top-ranked wrong candidate

Example:

- base: "Buy the least expensive microwave under $100."
- distraction version: "Buy the least expensive microwave under $100. Make sure it is **not black**."

When to use:

- the wrong candidate is highly visible, high-ranked, or especially salient
- the forbidden attribute is visible or strongly reflected in the page content

Good trap properties:

- wrong item appears earlier than the correct item
- wrong item shares most task attributes except one disallowed property
- correct item remains available and identifiable

Main risk:

- if the negative constraint is only textual and not visually grounded, the task can degenerate into pure instruction-following rather than distraction

Use this template only when the forbidden attribute is perceptually meaningful:

- color
- visible pattern
- visible object count
- visible visual style


### Template B: Fine-grained Attribute Confusion

Definition:

- create two or more highly similar candidates that differ on a subtle but crucial attribute

Example:

- correct: "Python 3.10 is officially released!"
- trap: "What features do you want in Python 3.10?"

When to use:

- two candidates are semantically or visually close
- titles, thumbnails, colors, brands, or other cues are easy to confuse
- a shallow scan can easily produce the wrong choice

This is the **primary template** for Category 2.

Reasons:

- it naturally induces near-miss errors
- it exposes weak discrimination
- it often produces realistic recovery behavior
- it is highly aligned with process-quality evaluation


### Template C: Layout / Ad Trap

Definition:

- exploit native webpage recommendation blocks, sponsored items, or sidebar modules to tempt the agent off the main path

Example:

- target: buy a camera body
- trap: a highly visible recommended lens in a "Frequently Bought Together" block

When to use:

- the layout trap is stable across runs
- the distractor is clearly visible and genuinely tempting
- the distractor sits near the intended path and can pull attention away

This is a **supplementary template**, not the main template.

Reasons:

- strong ecological realism
- but often lower controllability and stability


## 7. Template Priority

Use the templates in this order:

1. `Template B: Fine-grained Attribute Confusion`
2. `Template A: Negative Constraint Trap`
3. `Template C: Layout / Ad Trap`

Recommended composition for the first full Category 2 set:

- 50% Template B
- 35% Template A
- 15% Template C


## 8. What Must Be Modified in Code

For every distraction task, modify the following:

### 8.1 `intent`

The new instruction must explicitly encode the trap-aware goal.

Requirement:

- the goal must still have a unique correct answer
- the trap should be easy to click but invalid under the modified intent


### 8.2 `evaluator`

This is mandatory.

The evaluator must:

- accept the correct target
- reject the trap target
- reject known near-miss targets

Examples:

- blacklist disallowed `item_id`
- require the exact correct `post_id`
- require both correct target and correct action outcome

If the evaluator is not patched, the task is invalid.


### 8.3 Optional environment/data patch

For Templates A and B:

- often enough to patch only `intent + evaluator`

For Template C:

- sometimes necessary to patch:
  - recommendation modules
  - ranking
  - candidate placement
  - promoted content visibility


## 9. Formal Validity Constraints

A distraction-augmented task can only enter Category 2 if all of the following hold:

- the task is still solvable
- the correct target still exists
- the correct target remains unique
- at least one trap target is highly similar to the correct target
- the trap target is at least as salient as the correct target, or appears earlier
- the evaluator cleanly separates correct vs trap targets
- the page behavior is stable across repeated runs


## 10. Rollout-Based Acceptance Criteria

A constructed task should not be admitted based only on intuition.

For each candidate distraction task, run rollouts on both:

- `base version`
- `distraction version`

Then compare statistics such as:

- wrong-click rate
- near-miss rate
- detour rate
- revisit rate
- excess step count over shortest successful path
- success drop relative to base version

The distraction version should show:

- clearly higher process difficulty than the base version
- more near-miss / detour behavior
- but **not** collapse to near-zero success

If the task becomes effectively impossible, reject it.


## 11. Data Schema for Category 2

Each accepted task should store:

- `base_task_id`
- `variant_id`
- `variant_type = "high_distraction"`
- `construction_template`
- `site`
- `correct_target_id`
- `trap_target_ids`
- `negative_constraints`
- `evaluator_patch_summary`
- `base_intent`
- `modified_intent`
- `difficulty_delta`
- `expected_failure_modes`
- `validation_stats`


## 12. What Not To Do

Do not:

- define Category 2 as "pages that look messy"
- use homepage-only screenshots to decide distraction validity
- accept tasks where the trap is unrelated noise
- accept tasks that become unsolvable
- use unpatched evaluators
- treat arbitrary long instructions as distraction


## 13. This Afternoon's Execution Plan

### Step 1: Select base tasks

Goal:

- choose an initial batch of high-quality base tasks

Recommended first batch:

- `shopping`: 20 tasks
- `classifieds`: 10 tasks
- `reddit`: optional 5 tasks

Selection preference:

- candidate-rich tasks
- visual-attribute tasks
- price/rank/choice tasks


### Step 2: Assign a construction template

For each selected base task:

- prefer Template B first
- if not feasible, use Template A
- use Template C only when the layout trap is clearly stable


### Step 3: Patch `intent`

Create a modified task instruction that encodes the trap.


### Step 4: Patch evaluator

For each modified task:

- define `correct_target_id`
- define `trap_target_ids`
- update evaluator logic accordingly


### Step 5: Sanity-check each task manually

Before rollout collection, verify:

- correct target exists
- trap target exists
- trap is tempting
- task still solvable
- evaluator rejects the trap


### Step 6: Collect rollouts

For each pair:

- run both base and distraction versions
- collect a small but meaningful rollout set

Suggested minimum:

- 3 agents or prompting settings
- 5 rollouts per setting


### Step 7: Keep only validated tasks

Keep a task only if the distraction version:

- increases near-miss/detour behavior
- preserves solvability
- gives a cleaner Category 2 signal than the base version


## 14. Final Recommendation

The official Category 2 construction method is:

> Build paired distraction-augmented tasks from VWA base tasks, primarily using fine-grained attribute confusion and negative-constraint traps, with layout traps as a controlled supplement; patch the evaluator to explicitly reject trap targets; and only keep tasks whose distraction effect is confirmed by rollout statistics.

This is the final recommended protocol for Web-PQB v2 Category 2.
