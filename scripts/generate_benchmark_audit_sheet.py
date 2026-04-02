#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_PATH = ROOT / "data" / "benchmark_tasks" / "webpqb_benchmark.json"
OUT_PATH = ROOT / "data" / "benchmark_tasks" / "webpqb_benchmark_audit.csv"


def summarize_eval(task: dict[str, Any]) -> tuple[str, str]:
    eval_cfg = task.get("eval", {}) or {}
    eval_types = eval_cfg.get("eval_types", []) or []
    eval_type_str = "|".join(eval_types)
    ref_url = eval_cfg.get("reference_url", "") or ""
    return eval_type_str, ref_url


def category_focus(category: str) -> str:
    if category == "multi_path":
        return "Confirm there are at least two natural paths; distinguish direct vs detour success."
    if category == "high_distraction":
        return "Confirm the page really contains distracting alternatives, not just a constraint-heavy goal."
    if category == "recovery":
        return "Confirm wrong_start is wrong-but-recoverable and recovery itself is non-trivial."
    return ""


def category_specific_defaults(task: dict[str, Any]) -> dict[str, str]:
    category = task.get("category", "")
    defaults = {
        "multi_path_confirmed": "",
        "distraction_visible": "",
        "recovery_setup_valid": "",
        "recovery_severity": "",
    }
    if category == "multi_path":
        defaults["multi_path_confirmed"] = "TODO"
    elif category == "high_distraction":
        defaults["distraction_visible"] = "TODO"
    elif category == "recovery":
        defaults["recovery_setup_valid"] = "TODO"
        defaults["recovery_severity"] = "TODO"
    return defaults


def build_row(task: dict[str, Any]) -> dict[str, Any]:
    eval_types, reference_url = summarize_eval(task)
    row: dict[str, Any] = {
        "benchmark_id": task.get("benchmark_id", ""),
        "category": task.get("category", ""),
        "site": task.get("site", ""),
        "source_benchmark": "visualwebarena",
        "task_id": task.get("task_id", ""),
        "source_task_id": task.get("source_task_id", ""),
        "overall_difficulty": task.get("overall_difficulty", ""),
        "intent": task.get("intent", ""),
        "start_url": task.get("start_url", ""),
        "wrong_start_url": task.get("wrong_start_url", ""),
        "original_start_url": task.get("original_start_url", ""),
        "require_login": task.get("require_login", ""),
        "require_reset": task.get("require_reset", ""),
        "eval_types": eval_types,
        "reference_url": reference_url,
        "audit_focus": category_focus(task.get("category", "")),
        "audit_status": "TODO",
        "keep_or_drop": "",
        "category_correct": "",
        "task_solvable": "",
        "evaluator_reliable": "",
        "site_balance_priority": "",
        "duplicate_or_near_duplicate": "",
        "visual_dependence": "",
        "notes": "",
        "audited_by": "",
        "review_round": "v1",
    }
    row.update(category_specific_defaults(task))
    return row


def main() -> None:
    tasks = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    rows = [build_row(task) for task in tasks]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "benchmark_id",
        "category",
        "site",
        "source_benchmark",
        "task_id",
        "source_task_id",
        "overall_difficulty",
        "intent",
        "start_url",
        "wrong_start_url",
        "original_start_url",
        "require_login",
        "require_reset",
        "eval_types",
        "reference_url",
        "audit_focus",
        "audit_status",
        "keep_or_drop",
        "category_correct",
        "multi_path_confirmed",
        "distraction_visible",
        "recovery_setup_valid",
        "recovery_severity",
        "task_solvable",
        "evaluator_reliable",
        "site_balance_priority",
        "duplicate_or_near_duplicate",
        "visual_dependence",
        "notes",
        "audited_by",
        "review_round",
    ]

    with open(OUT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} audit rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
