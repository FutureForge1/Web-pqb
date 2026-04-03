#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from benchmark_v2_common import (
    BENCHMARK_V2_DIR,
    dump_json,
    high_distraction_candidate,
    infer_visual_dependence,
    load_vwa_tasks,
    write_jsonl,
)


DEFAULT_JSONL = BENCHMARK_V2_DIR / "native_high_distraction_candidates.jsonl"
DEFAULT_CSV = BENCHMARK_V2_DIR / "native_high_distraction_candidates.csv"
DEFAULT_SUMMARY = BENCHMARK_V2_DIR / "native_high_distraction_candidates_summary.json"

NEGATIVE_PATTERNS = [
    r"\bnot\b",
    r"\bcannot\b",
    r"\bcan't\b",
    r"\bdo not\b",
    r"\bdon't\b",
    r"\bwithout\b",
    r"\bexcept\b",
    r"\brather than\b",
]

IMAGE_PATTERNS = [
    r"\bimage\b",
    r"\bpicture\b",
    r"\bphoto\b",
    r"\bshown\b",
    r"\bpictured\b",
]

RANK_PATTERNS = [
    r"\bcheapest\b",
    r"\bleast expensive\b",
    r"\bmost expensive\b",
    r"\bmost recently\b",
    r"\blatest\b",
]

SIMILARITY_PATTERNS = [
    r"\bsame brand\b",
    r"\bsame as\b",
    r"\blooks like\b",
    r"\bresembles\b",
    r"\bsimilar\b",
    r"\bclosely resembles\b",
]

VISUAL_ATTRIBUTE_PATTERNS = [
    r"\bred\b",
    r"\bblue\b",
    r"\bblack\b",
    r"\bwhite\b",
    r"\bgreen\b",
    r"\byellow\b",
    r"\bpurple\b",
    r"\bpink\b",
    r"\bbrown\b",
    r"\bsilver\b",
    r"\bgold\b",
    r"\bpattern\b",
    r"\bdesign\b",
    r"\blogo\b",
    r"\bposter\b",
    r"\bprint\b",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract native high-distraction-like tasks from raw VWA tasks.")
    parser.add_argument("--output-jsonl", default=str(DEFAULT_JSONL))
    parser.add_argument("--output-csv", default=str(DEFAULT_CSV))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--min-score", type=int, default=5)
    parser.add_argument("--site", choices=["shopping", "classifieds", "reddit", "all"], default="all")
    return parser.parse_args()


def matches_any(text: str, patterns: list[str]) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in patterns)


def classify_candidate(task: dict) -> dict | None:
    intent = task.get("intent", "")
    lowered = intent.lower()

    score = 0
    reasons: list[str] = []

    negative = matches_any(lowered, NEGATIVE_PATTERNS)
    image = matches_any(lowered, IMAGE_PATTERNS)
    rank = matches_any(lowered, RANK_PATTERNS)
    similarity = matches_any(lowered, SIMILARITY_PATTERNS)
    visual_attr = matches_any(lowered, VISUAL_ATTRIBUTE_PATTERNS)

    if negative:
        score += 3
        reasons.append("带负面约束")
    if image:
        score += 2
        reasons.append("有图片/视觉线索")
    if rank:
        score += 2
        reasons.append("有排序目标")
    if similarity:
        score += 2
        reasons.append("有相似性约束")
    if visual_attr:
        score += 1
        reasons.append("有细粒度视觉属性")
    if high_distraction_candidate(task):
        score += 1
        reasons.append("原启发式命中过高干扰")

    if negative and rank and visual_attr:
        template = "负面约束陷阱"
    elif similarity and (image or visual_attr):
        template = "细粒度属性混淆"
    elif rank and image and visual_attr:
        template = "细粒度属性混淆"
    else:
        template = "待人工判断"

    if score < 1:
        return None

    return {
        "task_card_id": f"vwa:{task['site']}:{task['task_id']}",
        "site": task["site"],
        "task_id": task["task_id"],
        "intent": intent,
        "score": score,
        "recommended_template": template,
        "reasons": reasons,
        "visual_dependence_prior": infer_visual_dependence(intent, task.get("visual_difficulty")),
        "overall_difficulty": task.get("overall_difficulty", ""),
        "visual_difficulty": task.get("visual_difficulty", ""),
        "reasoning_difficulty": task.get("reasoning_difficulty", ""),
        "start_url": task.get("start_url", ""),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task_card_id",
        "site",
        "task_id",
        "score",
        "recommended_template",
        "visual_dependence_prior",
        "overall_difficulty",
        "visual_difficulty",
        "reasoning_difficulty",
        "intent",
        "reasons",
        "start_url",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row = dict(row)
            row["reasons"] = "；".join(row.get("reasons", []))
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    tasks = load_vwa_tasks()
    if args.site != "all":
        tasks = [task for task in tasks if task["site"] == args.site]

    candidates = []
    for task in tasks:
        row = classify_candidate(task)
        if row is None:
            continue
        if row["score"] >= args.min_score:
            candidates.append(row)

    candidates.sort(key=lambda row: (-row["score"], row["site"], int(row["task_id"])))

    write_jsonl(Path(args.output_jsonl), candidates)
    write_csv(Path(args.output_csv), candidates)

    summary = {
        "num_candidates": len(candidates),
        "site_distribution": {},
        "template_distribution": {},
    }
    for row in candidates:
        summary["site_distribution"][row["site"]] = summary["site_distribution"].get(row["site"], 0) + 1
        template = row["recommended_template"]
        summary["template_distribution"][template] = summary["template_distribution"].get(template, 0) + 1
    dump_json(Path(args.summary), summary)

    print(f"Wrote {len(candidates)} candidates -> {args.output_jsonl}")
    print(f"CSV -> {args.output_csv}")
    print(f"Summary -> {args.summary}")


if __name__ == "__main__":
    main()
