#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from benchmark_v2_common import (
    BENCHMARK_V2_DIR,
    count_constraints,
    dump_json,
    high_distraction_candidate,
    infer_visual_dependence,
    load_vwa_tasks,
    multi_path_candidate,
    parse_eval_types,
    recovery_base_candidate,
    summarize_by_key,
    task_card_id,
    write_jsonl,
)


DEFAULT_OUTPUT = BENCHMARK_V2_DIR / "vwa_task_cards.jsonl"
DEFAULT_SUMMARY = BENCHMARK_V2_DIR / "vwa_task_cards_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Web-PQB v2 task cards from VWA raw tasks.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    return parser.parse_args()


def build_task_card(task: dict) -> dict:
    candidates = []
    if multi_path_candidate(task):
        candidates.append("multi_path")
    if high_distraction_candidate(task):
        candidates.append("high_distraction")
    if recovery_base_candidate(task):
        candidates.append("recovery_base")
    if not candidates:
        candidates.append("other")

    return {
        "task_card_id": task_card_id(task["site"], task["task_id"]),
        "source_benchmark": "visualwebarena",
        "source_split": "test",
        "site": task["site"],
        "task_id": task["task_id"],
        "intent": task.get("intent", ""),
        "intent_template": task.get("intent_template", ""),
        "start_url": task.get("start_url", ""),
        "storage_state": task.get("storage_state", ""),
        "require_login": bool(task.get("require_login", False)),
        "require_reset": bool(task.get("require_reset", False)),
        "overall_difficulty": task.get("overall_difficulty", ""),
        "reasoning_difficulty": task.get("reasoning_difficulty", ""),
        "visual_difficulty": task.get("visual_difficulty", ""),
        "visual_dependence_prior": infer_visual_dependence(
            task.get("intent", ""),
            task.get("visual_difficulty"),
        ),
        "constraint_count": count_constraints(task.get("intent", "")),
        "category_candidates": candidates,
        "eval": task.get("eval", {}),
        "eval_types": parse_eval_types(task.get("eval", {})),
        "instantiation_dict": task.get("instantiation_dict", {}),
        "image_reference": task.get("image", ""),
        "comments": task.get("comments", ""),
    }


def main() -> None:
    args = parse_args()
    tasks = load_vwa_tasks()
    cards = [build_task_card(task) for task in tasks]

    output_path = Path(args.output)
    summary_path = Path(args.summary)

    write_jsonl(output_path, cards)

    summary = {
        "num_task_cards": len(cards),
        "site_distribution": summarize_by_key(cards, "site"),
        "visual_dependence_prior": summarize_by_key(cards, "visual_dependence_prior"),
        "overall_difficulty": summarize_by_key(cards, "overall_difficulty"),
        "candidate_counts": {
            label: sum(label in card["category_candidates"] for card in cards)
            for label in ["multi_path", "high_distraction", "recovery_base", "other"]
        },
    }
    dump_json(summary_path, summary)

    print(f"Wrote {len(cards)} task cards -> {output_path}")
    print(f"Summary -> {summary_path}")


if __name__ == "__main__":
    main()
