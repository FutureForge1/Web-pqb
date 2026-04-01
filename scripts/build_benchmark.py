"""
Web-PQB Benchmark Final Task Set Builder

从三类任务里各取 60 条，构成 180 条核心 benchmark 集。
输出：data/benchmark_tasks/webpqb_benchmark.json
"""

import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "benchmark_tasks"

random.seed(42)

HOME_URLS = {
    "shopping": {"http://localhost:7770", "http://localhost:7770/"},
    "classifieds": {"http://localhost:9980"},
    "reddit": {"http://localhost:9999", "http://localhost:9999/forums/all"},
}

CONSTRAINT_PATTERNS = [
    r"\b(black|white|red|blue|green|silver|gold|gray|grey|pink|purple|brown|orange)\b",
    r"\b(under|less than|below|more than|above|over|between)\s+\d+",
    r"\bnot (refurbished|used|new|completely)\b",
    r"\b(from|in|located in)\s+[A-Z][a-z]+",
    r"\bbut (not|do not)\b",
    r"\bonly if\b",
    r"\bexcept\b|\bexclude\b",
    r"\d+gb|\d+tb|\d+\s*inch|\d+\s*pound",
    r"\b(brand new|refurbished)\b",
    r"\b(same brand|same model|same type)\b",
]

MULTI_PATH_TEMPLATES = [
    r"find.{0,30}(cheapest|most expensive|least expensive|lowest price)",
    r"find me.{0,30}(product|item)",
    r"navigate to the.{0,30}(most|least|cheapest|expensive)",
    r"what is the (price|cost|color|name|number|weight)",
    r"tell me.{0,30}(price|cost|how much|name|number)",
    r"list.{0,30}(product|item|name)",
    r"search for.{0,50}navigate",
]


def load_tasks(site):
    with open(ROOT / "data" / "vwa" / f"test_{site}.json") as f:
        tasks = json.load(f)
    for t in tasks:
        t["site"] = site
    return tasks


def is_from_homepage(t):
    return t["start_url"] in HOME_URLS[t["site"]] or "|AND|" in t["start_url"]


def count_constraints(intent):
    return sum(1 for p in CONSTRAINT_PATTERNS if re.search(p, intent, re.I))


def is_high_distraction(t):
    return is_from_homepage(t) and count_constraints(t["intent"]) >= 2


def is_multi_path(t):
    return is_from_homepage(t) and any(re.search(p, t["intent"], re.I) for p in MULTI_PATH_TEMPLATES)



def main():
    all_tasks = []
    for site in ["shopping", "classifieds", "reddit"]:
        all_tasks.extend(load_tasks(site))

    def balanced_sample(tasks, n=60):
        """按难度均衡采样 n 条：hard/medium/easy 各 1/3，不足则补充其他难度"""
        by_diff = {"hard": [], "medium": [], "easy": []}
        for t in tasks:
            d = t.get("overall_difficulty", "medium")
            by_diff.setdefault(d, []).append(t)
        per_bucket = n // 3
        selected = []
        remainder = []
        for d in ["hard", "medium", "easy"]:
            bucket = by_diff.get(d, [])
            random.shuffle(bucket)
            selected.extend(bucket[:per_bucket])
            remainder.extend(bucket[per_bucket:])
        # 补足到 n
        random.shuffle(remainder)
        selected.extend(remainder[: n - len(selected)])
        return selected[:n]

    # ── Category 1: Multi-path Convergence ───────────────────────────────────
    cat1 = [t for t in all_tasks if is_multi_path(t) and not is_high_distraction(t)]
    cat1_selected = balanced_sample(cat1, 60)

    # ── Category 2: High-distraction Trap ────────────────────────────────────
    cat2 = [t for t in all_tasks if is_high_distraction(t)]
    cat2_selected = balanced_sample(cat2, min(60, len(cat2)))

    # ── Category 3: Recovery-centric ─────────────────────────────────────────
    with open(OUT_DIR / "recovery_tasks.json") as f:
        cat3_all = json.load(f)
    cat3_selected = balanced_sample(cat3_all, 60)

    # ── 合并输出 ──────────────────────────────────────────────────────────────
    benchmark = []
    for i, t in enumerate(cat1_selected):
        benchmark.append({
            "benchmark_id": f"webpqb_{i+1:03d}",
            "category": "multi_path",
            "site": t["site"],
            "task_id": t["task_id"],
            "intent": t["intent"],
            "start_url": t["start_url"],
            "storage_state": t["storage_state"],
            "require_login": t["require_login"],
            "require_reset": t["require_reset"],
            "eval": t["eval"],
            "overall_difficulty": t.get("overall_difficulty", ""),
        })
    for i, t in enumerate(cat2_selected):
        benchmark.append({
            "benchmark_id": f"webpqb_{i+61:03d}",
            "category": "high_distraction",
            "site": t["site"],
            "task_id": t["task_id"],
            "intent": t["intent"],
            "start_url": t["start_url"],
            "storage_state": t["storage_state"],
            "require_login": t["require_login"],
            "require_reset": t["require_reset"],
            "eval": t["eval"],
            "overall_difficulty": t.get("overall_difficulty", ""),
        })
    for i, t in enumerate(cat3_selected):
        benchmark.append({
            "benchmark_id": f"webpqb_{i+121:03d}",
            "category": "recovery",
            "site": t["site"],
            "source_task_id": t.get("source_task_id", ""),
            "intent": t["intent"],
            "start_url": t["start_url"],
            "wrong_start_url": t["wrong_start_url"],
            "original_start_url": t["original_start_url"],
            "storage_state": t["storage_state"],
            "require_login": t["require_login"],
            "require_reset": t["require_reset"],
            "eval": t["eval"],
            "overall_difficulty": t.get("overall_difficulty", ""),
            "construction_note": t.get("construction_note", ""),
        })

    out_path = OUT_DIR / "webpqb_benchmark.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, ensure_ascii=False, indent=2)

    # 统计
    print(f"=== Web-PQB Benchmark: {len(benchmark)} tasks ===")
    from collections import Counter
    cats = Counter(t["category"] for t in benchmark)
    sites = Counter(t["site"] for t in benchmark)
    diffs = Counter(t["overall_difficulty"] for t in benchmark)
    print(f"categories: {dict(cats)}")
    print(f"sites:      {dict(sites)}")
    print(f"difficulty: {dict(diffs)}")
    print(f"\n保存至: {out_path}")

    # 各类示例
    for cat in ["multi_path", "high_distraction", "recovery"]:
        sub = [t for t in benchmark if t["category"] == cat]
        print(f"\n--- {cat} 示例 ---")
        for t in sub[:3]:
            print(f"  [{t['site']}] {t['intent'][:90]}")
            if cat == "recovery":
                print(f"    wrong_start: {t['wrong_start_url']}")


if __name__ == "__main__":
    main()
