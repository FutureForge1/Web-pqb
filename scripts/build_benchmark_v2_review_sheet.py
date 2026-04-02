#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from benchmark_v2_common import BENCHMARK_V2_DIR, read_jsonl


DEFAULT_TASK_CARDS = BENCHMARK_V2_DIR / "task_cards_hard_filtered.jsonl"
DEFAULT_SCREENING = BENCHMARK_V2_DIR / "vlm_screening.jsonl"
DEFAULT_OUTPUT = BENCHMARK_V2_DIR / "benchmark_v2_review_sheet.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the final human review sheet for Web-PQB v2.")
    parser.add_argument("--task-cards", default=str(DEFAULT_TASK_CARDS))
    parser.add_argument("--screening", default=str(DEFAULT_SCREENING))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cards = read_jsonl(Path(args.task_cards))
    screening = {row["task_card_id"]: row for row in read_jsonl(Path(args.screening))}

    rows = []
    for card in cards:
        row = {
            "task_card_id": card.get("task_card_id"),
            "base_task_card_id": card.get("base_task_card_id"),
            "site": card.get("site"),
            "task_family": card.get("task_family"),
            "category_candidates": json.dumps(card.get("category_candidates", []), ensure_ascii=False),
            "intent": card.get("intent"),
            "start_url": card.get("start_url"),
            "original_start_url": card.get("original_start_url"),
            "wrong_start_url": card.get("wrong_start_url"),
            "visual_dependence_prior": card.get("visual_dependence_prior"),
            "constraint_count": card.get("constraint_count"),
            "overall_difficulty": card.get("overall_difficulty"),
            "hard_filter_pass": card.get("hard_filter_pass"),
            "hard_filter_failures": json.dumps(card.get("hard_filter_failures", []), ensure_ascii=False),
            "human_final_decision": "",
            "human_final_category": "",
            "human_final_severity": "",
            "human_notes": "",
        }
        pred = screening.get(card["task_card_id"], {})
        for key in [
            "parse_ok",
            "num_images",
            "category_supported",
            "visual_dependence",
            "multi_path_valid",
            "route_plurality",
            "distraction_visible",
            "distractor_density",
            "target_confusability",
            "recovery_wrong_start_valid",
            "recovery_recoverable",
            "recovery_misleadingness",
            "recovery_answer_leakage",
            "recovery_severity",
            "confidence",
            "triage_label",
            "rationale_short",
        ]:
            row[f"vlm_{key}"] = pred.get(key)
        rows.append(row)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Review sheet -> {output_path}")


if __name__ == "__main__":
    main()
