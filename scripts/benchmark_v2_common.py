#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent.parent
VWA_DIR = ROOT / "data" / "vwa"
BENCHMARK_V2_DIR = ROOT / "data" / "benchmark_tasks_v2"
BENCHMARK_V2_DIR.mkdir(parents=True, exist_ok=True)

SITES = ("shopping", "classifieds", "reddit")

SITE_HOSTS = {
    "shopping": {"localhost:7770"},
    "classifieds": {"localhost:9980"},
    "reddit": {"localhost:9999"},
}

HOME_URLS = {
    "shopping": {"http://localhost:7770", "http://localhost:7770/"},
    "classifieds": {"http://localhost:9980", "http://localhost:9980/"},
    "reddit": {"http://localhost:9999", "http://localhost:9999/forums/all", "http://localhost:9999/"},
}

RECOVERY_WRONG_STARTS: dict[str, list[dict[str, str]]] = {
    "shopping": [
        {
            "url": "http://localhost:7770/grocery-gourmet-food/dairy-cheese-eggs/cheese.html",
            "candidate_type": "wrong_category",
            "severity_prior": "medium",
        },
        {
            "url": "http://localhost:7770/video-games/nintendo-switch.html",
            "candidate_type": "wrong_category",
            "severity_prior": "medium",
        },
        {
            "url": "http://localhost:7770/beauty-personal-care/oral-care/orthodontic-supplies.html",
            "candidate_type": "wrong_category",
            "severity_prior": "high",
        },
        {
            "url": "http://localhost:7770/home-kitchen/wall-art/posters-prints.html",
            "candidate_type": "wrong_category",
            "severity_prior": "medium",
        },
        {
            "url": "http://localhost:7770/office-products/office-electronics/printers-accessories.html?p=3",
            "candidate_type": "deep_listing_page",
            "severity_prior": "high",
        },
    ],
    "classifieds": [
        {
            "url": "http://localhost:9980/index.php?page=search&sCategory=18",
            "candidate_type": "wrong_search_results",
            "severity_prior": "medium",
        },
        {
            "url": "http://localhost:9980/index.php?page=search&sCategory=12&sShowAs=gallery",
            "candidate_type": "wrong_search_results",
            "severity_prior": "medium",
        },
        {
            "url": "http://localhost:9980/index.php?page=search&sCategory=10&iPage=8",
            "candidate_type": "deep_listing_page",
            "severity_prior": "high",
        },
        {
            "url": "http://localhost:9980/index.php?page=item&id=16826",
            "candidate_type": "wrong_item_detail",
            "severity_prior": "high",
        },
        {
            "url": "http://localhost:9980/index.php?page=item&id=74603",
            "candidate_type": "wrong_item_detail",
            "severity_prior": "high",
        },
    ],
    "reddit": [
        {
            "url": "http://localhost:9999/f/EarthPorn",
            "candidate_type": "wrong_forum",
            "severity_prior": "medium",
        },
        {
            "url": "http://localhost:9999/f/memes",
            "candidate_type": "wrong_forum",
            "severity_prior": "medium",
        },
        {
            "url": "http://localhost:9999/f/gaming",
            "candidate_type": "wrong_forum",
            "severity_prior": "medium",
        },
        {
            "url": "http://localhost:9999/f/memes/41674",
            "candidate_type": "wrong_post_detail",
            "severity_prior": "high",
        },
        {
            "url": "http://localhost:9999/f/food/18831",
            "candidate_type": "wrong_post_detail",
            "severity_prior": "high",
        },
    ],
}

MULTI_PATH_PATTERNS = [
    r"\b(find|search|look(?:ing)? for)\b.*\b(product|item|listing|post|comment|user|review)\b",
    r"\bnavigate to\b",
    r"\b(most|least|cheapest|lowest price|most expensive)\b",
    r"\bwhat is the (price|cost|number|name|title|color)\b",
    r"\btell me\b.*\b(price|cost|how much|name|number)\b",
]

CONSTRAINT_PATTERNS = [
    r"\b(black|white|red|blue|green|silver|gold|gray|grey|pink|purple|brown|orange)\b",
    r"\b(small|medium|large|xl|xxl|\d+gb|\d+tb|\d+\s*inch|\d+\")\b",
    r"\b(under|less than|below|more than|above|over|between)\s+\$?\d+",
    r"\bnot (refurbished|used|new|pre-owned)\b",
    r"\b(from|in|located in)\s+[A-Z][a-z]+\b",
    r"\b(brand|model|version|type|color|size|price|condition)\b",
    r"\b(only if|exclude|without|except)\b",
    r"\bbut (not|don't|do not)\b",
]

VISUAL_REFERENCE_PATTERNS = [
    r"\bimage\b",
    r"\bpicture\b",
    r"\bscreenshot\b",
    r"\bphoto\b",
    r"\bshown\b",
    r"\bin the image\b",
]


def load_vwa_tasks() -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for site in SITES:
        path = VWA_DIR / f"test_{site}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        for raw in payload:
            raw = dict(raw)
            raw["site"] = site
            tasks.append(raw)
    return tasks


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def task_card_id(site: str, task_id: Any) -> str:
    return f"vwa:{site}:{task_id}"


def parse_eval_types(eval_cfg: Any) -> list[str]:
    counter: Counter[str] = Counter()

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "eval_types" and isinstance(value, list):
                    for item in value:
                        counter[str(item)] += 1
                if key in {"eval_type", "type"} and isinstance(value, str):
                    counter[value] += 1
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(eval_cfg)
    return sorted(counter)


def is_homepage_url(site: str, url: str) -> bool:
    return url in HOME_URLS.get(site, set()) or "|AND|" in url


def count_constraints(text: str) -> int:
    lowered = text.lower()
    return sum(1 for pattern in CONSTRAINT_PATTERNS if re.search(pattern, lowered))


def matches_any(text: str, patterns: list[str]) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in patterns)


def infer_visual_dependence(intent: str, visual_difficulty: str | None) -> str:
    if matches_any(intent, VISUAL_REFERENCE_PATTERNS):
        return "high"
    if (visual_difficulty or "").lower() == "hard":
        return "high"
    if (visual_difficulty or "").lower() == "medium":
        return "medium"
    return "low"


def multi_path_candidate(task: dict[str, Any]) -> bool:
    intent = task.get("intent", "")
    return is_homepage_url(task["site"], task.get("start_url", "")) and matches_any(intent, MULTI_PATH_PATTERNS)


def high_distraction_candidate(task: dict[str, Any]) -> bool:
    intent = task.get("intent", "")
    return count_constraints(intent) >= 2 or infer_visual_dependence(intent, task.get("visual_difficulty")) == "high"


def recovery_base_candidate(task: dict[str, Any]) -> bool:
    return is_homepage_url(task["site"], task.get("start_url", ""))


def site_matches_url(site: str, url: str | None) -> bool:
    if not url:
        return False
    parsed = urlparse(url)
    host = parsed.netloc
    return host in SITE_HOSTS.get(site, set())


def summarize_by_key(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counter = Counter(str(row.get(key, "")) for row in rows)
    return dict(sorted(counter.items()))
