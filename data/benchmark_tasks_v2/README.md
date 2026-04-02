# Web-PQB v2 Construction

This directory contains the task-centric benchmark-v2 pipeline artifacts.

Core idea:
- start from raw `VisualWebArena` tasks, not a single agent's trajectory
- apply objective hard constraints first
- collect real page evidence from `start_url` and, for recovery, `wrong_start_url`
- use a VLM for structured screening, not direct final curation
- reserve humans for boundary cases, recovery, and final sign-off

Recommended pipeline:
1. `python scripts/generate_benchmark_v2_task_cards.py`
2. `python scripts/generate_benchmark_v2_recovery_candidates.py`
3. `python scripts/filter_benchmark_v2_hard_constraints.py`
4. `python scripts/capture_benchmark_v2_pages.py`
5. `python scripts/screen_benchmark_v2_with_vlm.py`
6. `python scripts/build_benchmark_v2_review_sheet.py`

Expected main artifacts:
- `vwa_task_cards.jsonl`: unified task-card layer from raw VWA tasks
- `recovery_task_cards.jsonl`: multiple wrong-start variants per base task
- `task_cards_hard_filtered.jsonl`: candidates that pass hard constraints
- `page_capture_manifest.jsonl`: real page-evidence capture log
- `vlm_screening.jsonl`: structured VLM screening fields + triage label
- `benchmark_v2_review_sheet.csv`: final human-review sheet

Three category families:
- `multi_path`: task supports multiple natural routes or candidate-selection paths
- `high_distraction`: page contains strong visual distractors or dense similar candidates
- `recovery`: wrong start is wrong-but-recoverable and recovery is itself the challenge
