"""
Web-PQB Benchmark Task Classifier
将 vwa_gpt4v_som 轨迹按三类任务族自动分类：
  1. multi_path   — 多路径收敛
  2. high_distraction — 高干扰陷阱
  3. recovery     — 逆风开局（需手工构造初始状态，此处仅标记候选）
"""

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CANONICAL_DIR = ROOT / "data" / "canonical"
OUT_DIR = ROOT / "data" / "benchmark_tasks"
OUT_DIR.mkdir(exist_ok=True)

# ── 关键词规则 ────────────────────────────────────────────────────────────────

# 类目一：多路径收敛
# 目标可通过搜索 OR 分类浏览 OR 用户列表等多条路径到达
MULTI_PATH_PATTERNS = [
    r"\bfind\b.*(product|item|listing|post|comment|user|review)",
    r"\bsearch\b.*(for|and)",
    r"\bnavigate to\b",
    r"\blook(ing)? for\b",
    r"\bmost (expensive|recent|popular|cheapest)\b",
    r"\bwhat is the (price|cost|number|name|title)\b",
    r"\btell me\b.*(price|cost|how much|name)",
]

# 类目二：高干扰陷阱
# goal 里包含 2+ 个约束条件（颜色/尺码/品牌/价格/条件/地区等）
CONSTRAINT_PATTERNS = [
    r"\b(black|white|red|blue|green|silver|gold|gray|grey|pink|purple|brown)\b",
    r"\b(small|medium|large|xl|xxl|[sml]-size|\d+gb|\d+tb|\d+\" |\d+inch)\b",
    r"\b(under|less than|below|more than|above|over|between)\s+\$?\d+",
    r"\bnot (refurbished|used|new)\b",
    r"\b(from|in|located in)\s+[A-Z][a-z]+",  # 地区约束
    r"\b(brand|model|version|type|color|size|price|condition)\b",
    r"\bonly\b|\bexclude\b|\bwithout\b|\bexcept\b",
    r"\band (also|additionally|furthermore)\b",
    r"\bbut (not|don't|do not)\b",
]

# 类目三：逆风开局候选
# 任务本身需要从某个特定状态开始，或明确涉及纠错/恢复
RECOVERY_PATTERNS = [
    r"\bon this page\b",          # 已在某个页面，需要识别当前状态
    r"\bthis (item|product|post|listing|image)\b",  # 指代当前页面内容
    r"\bnavigate (back|away|to another)\b",
    r"\bgo back\b",
    r"\bcorrect\b|\bfix\b|\bundo\b",
    r"\bwrong\b|\bmistake\b",
]


def count_constraints(goal: str) -> int:
    goal_lower = goal.lower()
    return sum(1 for p in CONSTRAINT_PATTERNS if re.search(p, goal_lower))


def matches_any(goal: str, patterns: list[str]) -> bool:
    goal_lower = goal.lower()
    return any(re.search(p, goal_lower) for p in patterns)


def classify_goal(goal: str) -> list[str]:
    """返回该 goal 匹配的所有类目（可多标签）"""
    labels = []
    n_constraints = count_constraints(goal)

    if n_constraints >= 2:
        labels.append("high_distraction")

    if matches_any(goal, RECOVERY_PATTERNS):
        labels.append("recovery_candidate")

    if matches_any(goal, MULTI_PATH_PATTERNS) and "high_distraction" not in labels:
        labels.append("multi_path")

    if not labels:
        labels.append("other")

    return labels


def main():
    df_traj = pd.read_parquet(CANONICAL_DIR / "vwa_gpt4v_som_trajectories.parquet")
    df_steps = pd.read_parquet(
        CANONICAL_DIR / "vwa_gpt4v_som_steps.parquet",
        columns=["trajectory_id", "step_idx", "action_method"],
    )

    step_counts = df_steps.groupby("trajectory_id").size().rename("num_steps")
    go_back_counts = (
        df_steps[df_steps["action_method"] == "go_back"]
        .groupby("trajectory_id")
        .size()
        .rename("go_back_count")
    )

    df = df_traj.join(step_counts, on="trajectory_id").join(
        go_back_counts, on="trajectory_id"
    )
    df["go_back_count"] = df["go_back_count"].fillna(0).astype(int)

    df["categories"] = df["goal"].apply(classify_goal)
    df["primary_category"] = df["categories"].apply(lambda x: x[0])

    # 统计
    print("=== 分类结果 ===")
    from collections import Counter
    all_cats = [c for cats in df["categories"] for c in cats]
    for cat, cnt in Counter(all_cats).most_common():
        print(f"  {cat}: {cnt}")

    print()
    print("=== 各类目网站分布 ===")
    for cat in ["multi_path", "high_distraction", "recovery_candidate", "other"]:
        sub = df[df["categories"].apply(lambda x: cat in x)]
        print(f"\n{cat} (n={len(sub)}):")
        print("  websites:", sub["website"].value_counts().to_dict())
        print("  success rate:", f"{sub['final_success'].mean():.2%}" if sub["final_success"].notna().any() else "N/A")
        print("  avg steps:", f"{sub['num_steps'].mean():.1f}")

    # 输出每类的任务列表
    for cat in ["multi_path", "high_distraction", "recovery_candidate"]:
        sub = df[df["categories"].apply(lambda x: cat in x)][
            ["trajectory_id", "website", "goal", "final_success", "num_steps", "go_back_count", "categories"]
        ].copy()
        sub["categories"] = sub["categories"].apply(json.dumps)
        out_path = OUT_DIR / f"{cat}_tasks.csv"
        sub.to_csv(out_path, index=False)
        print(f"\n保存 {cat}: {out_path} ({len(sub)} 条)")

    # 汇总 JSON
    summary = []
    for _, row in df.iterrows():
        summary.append({
            "trajectory_id": row["trajectory_id"],
            "website": row["website"],
            "goal": row["goal"],
            "final_success": bool(row["final_success"]) if pd.notna(row["final_success"]) else None,
            "num_steps": int(row["num_steps"]) if pd.notna(row["num_steps"]) else None,
            "go_back_count": int(row["go_back_count"]),
            "categories": row["categories"],
            "primary_category": row["primary_category"],
        })
    with open(OUT_DIR / "all_classified.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n汇总保存: {OUT_DIR / 'all_classified.json'}")


if __name__ == "__main__":
    main()
