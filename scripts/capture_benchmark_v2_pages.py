#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmark_v2_common import BENCHMARK_V2_DIR, read_jsonl


DEFAULT_INPUT = BENCHMARK_V2_DIR / "task_cards_hard_filtered.jsonl"
DEFAULT_OUTPUT = BENCHMARK_V2_DIR / "page_capture_manifest.jsonl"
DEFAULT_SCREENSHOT_DIR = BENCHMARK_V2_DIR / "page_evidence"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture start/wrong-start page evidence for Web-PQB v2 task cards.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--screenshot-dir", default=str(DEFAULT_SCREENSHOT_DIR))
    parser.add_argument("--auth-root", default="", help="Optional directory that stores Playwright auth json files.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--viewport-width", type=int, default=1440)
    parser.add_argument("--viewport-height", type=int, default=1600)
    parser.add_argument("--timeout-ms", type=int, default=30000)
    return parser.parse_args()


def resolve_storage_state(storage_state: str, auth_root: str) -> str | None:
    if not storage_state:
        return None
    candidates = [
        Path(storage_state),
        Path.cwd() / storage_state,
        Path(__file__).resolve().parent.parent / storage_state,
    ]
    if auth_root:
        candidates.append(Path(auth_root) / Path(storage_state).name)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return None


def split_compound_url(raw_url: str | None) -> list[dict[str, str]]:
    if not raw_url:
        return []
    raw_url = raw_url.strip()
    if not raw_url:
        return []

    if " |AND| " in raw_url:
        operator = "AND"
        parts = [part.strip() for part in raw_url.split(" |AND| ") if part.strip()]
    elif " |OR| " in raw_url:
        operator = "OR"
        parts = [part.strip() for part in raw_url.split(" |OR| ") if part.strip()]
    else:
        operator = "SINGLE"
        parts = [raw_url]

    return [
        {
            "url": url,
            "compound_operator": operator,
            "compound_index": str(idx),
        }
        for idx, url in enumerate(parts, start=1)
    ]


def planned_pages(card: dict[str, Any]) -> list[dict[str, str]]:
    pages: list[dict[str, str]] = []
    for target in split_compound_url(card.get("start_url", "")):
        pages.append({"page_role": "start", **target})
    if card.get("task_family") == "recovery":
        if card.get("original_start_url"):
            for target in split_compound_url(card["original_start_url"]):
                pages.append({"page_role": "original_start", **target})
        if card.get("wrong_start_url") and card["wrong_start_url"] != card.get("start_url"):
            for target in split_compound_url(card["wrong_start_url"]):
                pages.append({"page_role": "wrong_start", **target})
    return pages


def load_done_keys(path: Path) -> set[tuple[str, str]]:
    done = set()
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            page_key = str(rec.get("page_key") or rec.get("page_role"))
            key = (str(rec.get("task_card_id")), page_key)
            done.add(key)
    return done


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    cards = read_jsonl(Path(args.input))
    if args.limit is not None:
        cards = cards[: args.limit]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    screenshot_dir = Path(args.screenshot_dir)
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    done_keys = load_done_keys(output_path) if args.resume else set()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise SystemExit(
            "Playwright is required for page capture. Install it with `pip install playwright` "
            "and run `playwright install chromium`."
        ) from exc

    total = len(cards)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless)
        for idx, card in enumerate(cards, start=1):
            storage_state = resolve_storage_state(card.get("storage_state", ""), args.auth_root)
            context_kwargs = {
                "viewport": {"width": args.viewport_width, "height": args.viewport_height},
            }
            if storage_state:
                context_kwargs["storage_state"] = storage_state
            context = browser.new_context(**context_kwargs)

            for target in planned_pages(card):
                page_key = f"{target['page_role']}__{target['compound_index']}"
                key = (card["task_card_id"], page_key)
                if key in done_keys:
                    continue
                screenshot_name = (
                    f"{card['task_card_id'].replace(':', '__')}__"
                    f"{target['page_role']}__{target['compound_index']}.png"
                )
                screenshot_path = screenshot_dir / screenshot_name
                record = {
                    "task_card_id": card["task_card_id"],
                    "base_task_card_id": card.get("base_task_card_id"),
                    "site": card.get("site"),
                    "task_family": card.get("task_family"),
                    "category_target": card.get("category_target", ""),
                    "page_role": target["page_role"],
                    "page_key": page_key,
                    "compound_operator": target["compound_operator"],
                    "compound_index": int(target["compound_index"]),
                    "requested_url": target["url"],
                    "storage_state_resolved": storage_state,
                    "screenshot_path": str(screenshot_path),
                    "status": "error",
                    "timestamp": datetime.now().isoformat(),
                }
                page = context.new_page()
                try:
                    response = page.goto(target["url"], wait_until="networkidle", timeout=args.timeout_ms)
                    page.screenshot(path=str(screenshot_path), full_page=False)
                    record.update(
                        {
                            "status": "ok",
                            "http_status": None if response is None else response.status,
                            "final_url": page.url,
                            "page_title": page.title(),
                        }
                    )
                except Exception as exc:  # pragma: no cover - depends on runtime env
                    record["error"] = repr(exc)
                finally:
                    page.close()
                append_jsonl(output_path, record)
                print(
                    f"[{idx}/{total}] {card['task_card_id']} {page_key} -> "
                    f"{record['status']}"
                )
            context.close()
        browser.close()

    print(f"Capture manifest -> {output_path}")


if __name__ == "__main__":
    main()
