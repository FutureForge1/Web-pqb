"""
Web-PQB Category 3: Recovery-centric Task Constructor

逻辑：
  从 multi_path / high_distraction 任务里取 goal，
  把 start_url 替换成同网站的一个"错误页面"，
  agent 需要先识别自己在错误位置，再恢复到正确路径完成任务。

错误页面选取原则：
  - 同网站，但与任务目标无关
  - shopping: 错误品类页 / 错误商品详情页
  - classifieds: 错误分类搜索页 / 错误商品详情页
  - reddit: 错误 subreddit 页
"""

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "benchmark_tasks"
OUT_DIR.mkdir(exist_ok=True)

random.seed(42)

# ── 各网站的"错误初始页"候选池 ────────────────────────────────────────────────
# 选取与常见任务目标无关的页面，确保 agent 需要主动导航才能完成任务

WRONG_STARTS = {
    "shopping": [
        # 错误品类页
        "http://localhost:7770/grocery-gourmet-food/dairy-cheese-eggs/cheese.html",
        "http://localhost:7770/video-games/nintendo-switch.html",
        "http://localhost:7770/beauty-personal-care/oral-care/orthodontic-supplies.html",
        "http://localhost:7770/home-kitchen/bedding/blankets-throws.html",
        "http://localhost:7770/electronics/video-projectors.html",
        "http://localhost:7770/grocery-gourmet-food/breads-bakery/cookies.html",
        "http://localhost:7770/home-kitchen/wall-art/posters-prints.html",
        "http://localhost:7770/home-kitchen/home-decor-products/artificial-plants-flowers.html",
        "http://localhost:7770/clothing-shoes-jewelry/women/uniforms-work-safety.html",
        "http://localhost:7770/office-products/office-electronics/printers-accessories.html?p=3",
    ],
    "classifieds": [
        # 错误分类搜索页
        "http://localhost:9980/index.php?page=search&sCategory=18",   # 某个无关分类
        "http://localhost:9980/index.php?page=search&sCategory=12&sShowAs=gallery",
        "http://localhost:9980/index.php?page=search&sCategory=9&iPage=2&sShowAs=gallery",
        "http://localhost:9980/index.php?page=search&sCategory=10&iPage=8",
        # 错误商品详情页
        "http://localhost:9980/index.php?page=item&id=16826",
        "http://localhost:9980/index.php?page=item&id=74603",
        "http://localhost:9980/index.php?page=item&id=48575",
    ],
    "reddit": [
        # 错误 subreddit
        "http://localhost:9999/f/EarthPorn",
        "http://localhost:9999/f/memes",
        "http://localhost:9999/f/gaming",
        "http://localhost:9999/f/movies",
        "http://localhost:9999/f/Music",
        "http://localhost:9999/f/headphones",
        # 错误帖子页
        "http://localhost:9999/f/memes/41674",
        "http://localhost:9999/f/food/18831",
    ],
}

# 正确首页（用于判断原任务是否从首页出发）
HOME_URLS = {
    "shopping": {"http://localhost:7770", "http://localhost:7770/"},
    "classifieds": {"http://localhost:9980"},
    "reddit": {"http://localhost:9999", "http://localhost:9999/forums/all"},
}


def load_tasks(site: str) -> list[dict]:
    with open(ROOT / "data" / "vwa" / f"test_{site}.json") as f:
        tasks = json.load(f)
    for t in tasks:
        t["site"] = site
    return tasks


def is_from_homepage(task: dict) -> bool:
    site = task["site"]
    return task["start_url"] in HOME_URLS[site] or "|AND|" in task["start_url"]


def pick_wrong_start(site: str, original_url: str) -> str:
    """从错误页候选池里随机选一个，避免和原始 URL 相同"""
    candidates = [u for u in WRONG_STARTS[site] if u != original_url]
    return random.choice(candidates)


def main():
    all_source_tasks = []
    for site in ["shopping", "classifieds", "reddit"]:
        all_source_tasks.extend(load_tasks(site))

    # 只取从首页出发的任务作为 source（这样替换 start_url 才有意义）
    homepage_tasks = [t for t in all_source_tasks if is_from_homepage(t)]
    print(f"首页出发任务总数: {len(homepage_tasks)}")

    # 按网站分组，各取一批构造 recovery 任务
    # 目标：每个网站 ~40 条，共 ~120 条
    TARGET_PER_SITE = 40
    recovery_tasks = []

    for site in ["shopping", "classifieds", "reddit"]:
        site_tasks = [t for t in homepage_tasks if t["site"] == site]
        # 优先选 hard/medium 难度
        site_tasks_sorted = sorted(
            site_tasks,
            key=lambda t: {"hard": 0, "medium": 1, "easy": 2}.get(t.get("overall_difficulty", "medium"), 1),
        )
        selected = site_tasks_sorted[:TARGET_PER_SITE]

        for t in selected:
            wrong_url = pick_wrong_start(site, t["start_url"])
            recovery_task = {
                "task_id": f"recovery_{t['task_id']}",
                "source_task_id": t["task_id"],
                "site": site,
                "sites": t["sites"],
                "intent": t["intent"],
                "intent_template": t.get("intent_template", ""),
                "original_start_url": t["start_url"],
                "start_url": wrong_url,          # 替换为错误页面
                "wrong_start_url": wrong_url,
                "storage_state": t["storage_state"],
                "require_login": t["require_login"],
                "require_reset": t["require_reset"],
                "eval": t["eval"],
                "overall_difficulty": t.get("overall_difficulty", ""),
                "reasoning_difficulty": t.get("reasoning_difficulty", ""),
                "visual_difficulty": t.get("visual_difficulty", ""),
                "category": "recovery",
                "construction_note": (
                    f"Original task starts at '{t['start_url']}'. "
                    f"Wrong start injected: '{wrong_url}'. "
                    "Agent must recognize wrong state and navigate to correct starting point."
                ),
            }
            recovery_tasks.append(recovery_task)

        print(f"  {site}: {len(selected)} recovery tasks constructed")

    # 保存
    out_path = OUT_DIR / "recovery_tasks.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(recovery_tasks, f, ensure_ascii=False, indent=2)

    print(f"\n总计: {len(recovery_tasks)} recovery tasks → {out_path}")

    # 打印几个示例
    print("\n=== 示例 ===")
    for t in recovery_tasks[:3]:
        print(f"[{t['site']}] goal: {t['intent'][:80]}")
        print(f"  wrong_start: {t['wrong_start_url']}")
        print(f"  original:    {t['original_start_url']}")
        print()


if __name__ == "__main__":
    main()
