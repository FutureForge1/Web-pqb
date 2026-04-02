#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from benchmark_v2_common import (
    BENCHMARK_V2_DIR,
    RECOVERY_WRONG_STARTS,
    dump_json,
    read_jsonl,
    site_matches_url,
    summarize_by_key,
    write_jsonl,
)


DEFAULT_INPUT = BENCHMARK_V2_DIR / "vwa_task_cards.jsonl"
DEFAULT_OUTPUT = BENCHMARK_V2_DIR / "recovery_task_cards.jsonl"
DEFAULT_SUMMARY = BENCHMARK_V2_DIR / "recovery_task_cards_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Web-PQB v2 recovery candidates with multiple wrong-start options.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--max-candidates-per-task", type=int, default=3)
    return parser.parse_args()


def build_recovery_variant(card: dict, candidate: dict, idx: int) -> dict:
    recovery_id = f"{card['task_card_id']}:recovery:{idx}"
    wrong_url = candidate["url"]
    return {
        **card,
        "task_card_id": recovery_id,
        "base_task_card_id": card["task_card_id"],
        "source_task_id": card["task_id"],
        "category_target": "recovery",
        "wrong_start_url": wrong_url,
        "original_start_url": card["start_url"],
        "start_url": wrong_url,
        "wrong_start_candidate_type": candidate["candidate_type"],
        "recovery_severity_prior": candidate["severity_prior"],
        "recovery_generation_note": (
            f"Generated from {card['task_card_id']} by replacing start_url with a same-site "
            f"{candidate['candidate_type']} wrong start."
        ),
        "recovery_hard_checks": {
            "wrong_start_differs_from_original": wrong_url != card["start_url"],
            "wrong_start_same_site": site_matches_url(card["site"], wrong_url),
        },
    }


def main() -> None:
    args = parse_args()
    cards = read_jsonl(Path(args.input))

    recovery_cards = []
    for card in cards:
        if "recovery_base" not in card.get("category_candidates", []):
            continue
        candidates = RECOVERY_WRONG_STARTS.get(card["site"], [])[: args.max_candidates_per_task]
        for idx, candidate in enumerate(candidates, start=1):
            recovery_cards.append(build_recovery_variant(card, candidate, idx))

    output_path = Path(args.output)
    summary_path = Path(args.summary)
    write_jsonl(output_path, recovery_cards)

    summary = {
        "num_recovery_variants": len(recovery_cards),
        "site_distribution": summarize_by_key(recovery_cards, "site"),
        "candidate_type_distribution": summarize_by_key(recovery_cards, "wrong_start_candidate_type"),
        "severity_prior_distribution": summarize_by_key(recovery_cards, "recovery_severity_prior"),
        "base_task_coverage": len({row["base_task_card_id"] for row in recovery_cards}),
    }
    dump_json(summary_path, summary)

    print(f"Wrote {len(recovery_cards)} recovery task cards -> {output_path}")
    print(f"Summary -> {summary_path}")


if __name__ == "__main__":
    main()
