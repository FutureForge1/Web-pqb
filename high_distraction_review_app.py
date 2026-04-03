from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components


ROOT = Path("/home/lenovo/code/NIPS2026")
DEFAULT_CANDIDATE_CSV = ROOT / "data" / "benchmark_tasks_v2" / "native_high_distraction_candidates_shopping.csv"
REVIEW_RECORDS = ROOT / "docs" / "benchmark_v2" / "shopping_high_distraction_review_records.jsonl"
REVIEW_URLS = ROOT / "docs" / "benchmark_v2" / "shopping_high_distraction_review_urls.json"
VWA_SHOPPING_DIR = ROOT / "data" / "vwa" / "test_shopping"


def load_candidates(path: Path) -> list[dict[str, Any]]:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    for row in rows:
        row["score"] = int(row.get("score") or 0)
        row["reasons_list"] = [part for part in (row.get("reasons") or "").split("；") if part]
    return rows


def load_task_json(task_card_id: str) -> dict[str, Any]:
    task_num = task_card_id.split(":")[-1]
    path = VWA_SHOPPING_DIR / f"{task_num}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_latest_reviews(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return latest
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            task_id = row.get("task_card_id")
            if task_id:
                latest[task_id] = row
    return latest


def append_review(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def homepage_like(url: str) -> bool:
    return url.rstrip("/") in {"http://localhost:7770", "http://localhost:9980", "http://localhost:9999"}


def default_review_url(candidate: dict[str, Any], url_map: dict[str, str], review_record: dict[str, Any] | None) -> str:
    task_id = candidate["task_card_id"]
    if review_record and review_record.get("review_url"):
        return str(review_record["review_url"])
    if task_id in url_map and url_map[task_id]:
        return str(url_map[task_id])
    start_url = str(candidate.get("start_url") or "")
    if start_url and not homepage_like(start_url):
        return start_url
    return ""


def review_status_label(review: dict[str, Any] | None) -> str:
    if not review:
        return "未审核"
    return str(review.get("fit_second_category") or "未审核")


def candidate_label(idx: int, row: dict[str, Any], review: dict[str, Any] | None) -> str:
    status = review_status_label(review)
    return f"{idx + 1}. {row['task_card_id']} | {row['recommended_template']} | {status}"


def short_text(text: str, limit: int = 72) -> str:
    text = text.strip().replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


st.set_page_config(page_title="第二类任务审核台", layout="wide")
st.title("第二类任务审核台")
st.caption("用于人工判断 shopping 的天然陷阱版候选，看看它们到底符不符合我们第二类的标准。")

candidates = load_candidates(DEFAULT_CANDIDATE_CSV)
latest_reviews = load_latest_reviews(REVIEW_RECORDS)
review_urls = load_json(REVIEW_URLS)

status_options = ["全部", "未审核", "符合", "不符合", "待定"]
template_options = ["全部"] + sorted({row["recommended_template"] for row in candidates})

with st.sidebar:
    st.header("筛选")
    status_filter = st.selectbox("审核状态", status_options, index=0)
    template_filter = st.selectbox("推荐模板", template_options, index=0)
    keyword = st.text_input("关键词", value="")

filtered = []
for row in candidates:
    review = latest_reviews.get(row["task_card_id"])
    status = review_status_label(review)
    if status_filter != "全部" and status != status_filter:
        continue
    if template_filter != "全部" and row["recommended_template"] != template_filter:
        continue
    blob = " ".join(
        [
            row["task_card_id"],
            row.get("intent", ""),
            row.get("recommended_template", ""),
            row.get("reasons", ""),
        ]
    ).lower()
    if keyword and keyword.lower() not in blob:
        continue
    filtered.append(row)

if not filtered:
    st.warning("当前筛选条件下没有任务。")
    st.stop()

if "review_index" not in st.session_state:
    st.session_state.review_index = 0

st.session_state.review_index = min(st.session_state.review_index, len(filtered) - 1)

with st.sidebar:
    st.write(f"当前候选数：{len(filtered)} / 总数 {len(candidates)}")
    selected_label = st.selectbox(
        "选择任务",
        [candidate_label(i, row, latest_reviews.get(row["task_card_id"])) for i, row in enumerate(filtered)],
        index=st.session_state.review_index,
    )
    st.session_state.review_index = [
        candidate_label(i, row, latest_reviews.get(row["task_card_id"])) for i, row in enumerate(filtered)
    ].index(selected_label)
    col_prev, col_next = st.columns(2)
    with col_prev:
        if st.button("上一题", use_container_width=True):
            st.session_state.review_index = max(0, st.session_state.review_index - 1)
            st.rerun()
    with col_next:
        if st.button("下一题", use_container_width=True):
            st.session_state.review_index = min(len(filtered) - 1, st.session_state.review_index + 1)
            st.rerun()

row = filtered[st.session_state.review_index]
review = latest_reviews.get(row["task_card_id"])
task = load_task_json(row["task_card_id"])

left, right = st.columns([1.1, 1.2])

with left:
    st.subheader(row["task_card_id"])
    st.write(f"推荐模板：`{row['recommended_template']}`")
    st.write(f"自动分数：`{row['score']}`")
    st.write("原始任务：")
    st.code(row["intent"], language="text")
    st.write("为什么它可能像第二类：")
    for reason in row["reasons_list"]:
        st.write(f"- {reason}")

    if task:
        with st.expander("看原始任务 JSON"):
            st.json(
                {
                    "task_id": task.get("task_id"),
                    "start_url": task.get("start_url"),
                    "intent": task.get("intent"),
                    "overall_difficulty": task.get("overall_difficulty"),
                    "visual_difficulty": task.get("visual_difficulty"),
                    "eval_types": task.get("eval", {}).get("eval_types", []),
                }
            )

    suggested_url = default_review_url(row, review_urls, review)
    review_url = st.text_input("挑选网页 URL", value=suggested_url, key=f"url_{row['task_card_id']}")

    col_save_url, col_open_url = st.columns([1, 1])
    with col_save_url:
        if st.button("保存这个网页地址", key=f"save_url_{row['task_card_id']}", use_container_width=True):
            review_urls[row["task_card_id"]] = review_url.strip()
            save_json(REVIEW_URLS, review_urls)
            st.success("网页地址已保存。")
    with col_open_url:
        if review_url:
            st.markdown(f"[新窗口打开这个页面]({review_url})")

    fit_second_category = st.radio(
        "这题符合第二类吗？",
        ["待定", "符合", "不符合"],
        index=["待定", "符合", "不符合"].index(review.get("fit_second_category", "待定") if review else "待定"),
        horizontal=True,
    )
    use_as_native_trap = st.radio(
        "这题能直接当“天然陷阱版”吗？",
        ["待定", "可以", "不可以"],
        index=["待定", "可以", "不可以"].index(review.get("use_as_native_trap", "待定") if review else "待定"),
        horizontal=True,
    )
    pair_strategy = st.selectbox(
        "原始版怎么处理？",
        ["待定", "补一个原始版", "已经天然有原始版", "不建议继续做"],
        index=["待定", "补一个原始版", "已经天然有原始版", "不建议继续做"].index(
            review.get("pair_strategy", "待定") if review else "待定"
        ),
    )
    notes = st.text_area("备注", value=review.get("notes", "") if review else "", height=220)

    if st.button("保存审核结果", type="primary", use_container_width=True):
        record = {
            "task_card_id": row["task_card_id"],
            "site": row["site"],
            "task_id": row["task_id"],
            "recommended_template": row["recommended_template"],
            "score": row["score"],
            "review_url": review_url.strip(),
            "fit_second_category": fit_second_category,
            "use_as_native_trap": use_as_native_trap,
            "pair_strategy": pair_strategy,
            "notes": notes,
            "timestamp": datetime.now().isoformat(),
        }
        append_review(REVIEW_RECORDS, record)
        latest_reviews[row["task_card_id"]] = record
        if review_url.strip():
            review_urls[row["task_card_id"]] = review_url.strip()
            save_json(REVIEW_URLS, review_urls)
        st.success("已保存。")

with right:
    st.subheader("网页预览")
    if review_url:
        components.html(
            f"""
            <iframe
                src="{review_url}"
                width="100%"
                height="920"
                style="border: 1px solid #ddd; border-radius: 8px;"
            ></iframe>
            """,
            height=940,
            scrolling=True,
        )
    else:
        st.info("这题目前还没有挑选网页 URL。你可以先在左边填一个，再回来审核。")

st.divider()
st.subheader("当前筛选结果概览")

overview_rows = []
for item in filtered:
    item_review = latest_reviews.get(item["task_card_id"], {})
    overview_rows.append(
        {
            "任务": item["task_card_id"],
            "模板": item["recommended_template"],
            "分数": item["score"],
            "状态": item_review.get("fit_second_category", "未审核"),
            "天然陷阱版": item_review.get("use_as_native_trap", "未审核"),
            "原始版处理": item_review.get("pair_strategy", "未审核"),
            "网页": default_review_url(item, review_urls, item_review) or "",
            "任务描述": short_text(item["intent"]),
        }
    )

st.dataframe(overview_rows, use_container_width=True, height=320)
