#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urlparse

from benchmark_v2_common import (
    BENCHMARK_V2_DIR,
    dump_json,
    read_jsonl,
    site_matches_url,
    summarize_by_key,
    write_jsonl,
)


DEFAULT_TASK_CARDS = BENCHMARK_V2_DIR / "vwa_task_cards.jsonl"
DEFAULT_RECOVERY_CARDS = BENCHMARK_V2_DIR / "recovery_task_cards.jsonl"
DEFAULT_FILTERED = BENCHMARK_V2_DIR / "task_cards_hard_filtered.jsonl"
DEFAULT_REPORT = BENCHMARK_V2_DIR / "hard_filter_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply hard constraints to Web-PQB v2 task cards.")
    parser.add_argument("--task-cards", default=str(DEFAULT_TASK_CARDS))
    parser.add_argument("--recovery-cards", default=str(DEFAULT_RECOVERY_CARDS))
    parser.add_argument("--output", default=str(DEFAULT_FILTERED))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    return parser.parse_args()


def url_is_http(url: str | None) -> bool:
    if not url:
        return False
    return urlparse(url).scheme in {"http", "https"}


def validate_common(card: dict) -> list[str]:
    failures: list[str] = []
    if not card.get("intent"):
        failures.append("missing_intent")
    if not url_is_http(card.get("start_url")):
        failures.append("invalid_start_url")
    if not site_matches_url(card.get("site", ""), card.get("start_url")):
        failures.append("start_url_site_mismatch")
    if not isinstance(card.get("eval"), dict) or not card.get("eval"):
        failures.append("missing_eval")
    if not card.get("storage_state"):
        failures.append("missing_storage_state")
    return failures


def validate_recovery(card: dict) -> list[str]:
    failures: list[str] = []
    wrong_start_url = card.get("wrong_start_url")
    original_start_url = card.get("original_start_url")
    if not url_is_http(wrong_start_url):
        failures.append("invalid_wrong_start_url")
    if not site_matches_url(card.get("site", ""), wrong_start_url):
        failures.append("wrong_start_site_mismatch")
    if wrong_start_url == original_start_url:
        failures.append("wrong_start_equals_original")
    checks = card.get("recovery_hard_checks", {})
    if checks.get("wrong_start_differs_from_original") is False:
        failures.append("wrong_start_equals_original")
    if checks.get("wrong_start_same_site") is False:
        failures.append("wrong_start_site_mismatch")
    return failures


def dedupe_key(card: dict) -> tuple[str, str, str]:
    return (
        str(card.get("site", "")),
        str(card.get("intent", "")).strip().lower(),
        str(card.get("start_url", "")).strip(),
    )


def main() -> None:
    args = parse_args()
    base_cards = read_jsonl(Path(args.task_cards))
    recovery_cards = read_jsonl(Path(args.recovery_cards))

    all_cards = []
    for row in base_cards:
        enriched = dict(row)
        enriched["task_family"] = "base"
        all_cards.append(enriched)
    for row in recovery_cards:
        enriched = dict(row)
        enriched["task_family"] = "recovery"
        all_cards.append(enriched)

    seen_keys: set[tuple[str, str, str]] = set()
    filtered_rows = []
    rejected_rows = []

    for card in all_cards:
        failures = validate_common(card)
        if card.get("task_family") == "recovery":
            failures.extend(validate_recovery(card))

        key = dedupe_key(card)
        if key in seen_keys:
            failures.append("duplicate_intent_start")
        else:
            seen_keys.add(key)

        out_row = dict(card)
        out_row["hard_filter_pass"] = not failures
        out_row["hard_filter_failures"] = failures
        if failures:
            rejected_rows.append(out_row)
        else:
            filtered_rows.append(out_row)

    output_path = Path(args.output)
    report_path = Path(args.report)
    write_jsonl(output_path, filtered_rows)

    failure_counts: dict[str, int] = {}
    for row in rejected_rows:
        for failure in row["hard_filter_failures"]:
            failure_counts[failure] = failure_counts.get(failure, 0) + 1

    report = {
        "num_input_rows": len(all_cards),
        "num_passed": len(filtered_rows),
        "num_rejected": len(rejected_rows),
        "passed_site_distribution": summarize_by_key(filtered_rows, "site"),
        "passed_task_family_distribution": summarize_by_key(filtered_rows, "task_family"),
        "failure_counts": dict(sorted(failure_counts.items())),
        "rejected_examples": rejected_rows[:20],
    }
    dump_json(report_path, report)

    print(f"Passed {len(filtered_rows)}/{len(all_cards)} rows -> {output_path}")
    print(f"Report -> {report_path}")


if __name__ == "__main__":
    main()
