from __future__ import annotations

import io
import json
import tarfile
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

from vwa_render_parser import decode_data_url_image, parse_render_pages


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CANONICAL_DIR = DATA_DIR / "canonical"

CANONICAL_SOURCES = {
    "mind2web": {
        "label": "Multimodal Mind2Web",
        "steps_path": CANONICAL_DIR / "mind2web_steps.parquet",
        "trajectories_path": CANONICAL_DIR / "mind2web_trajectories.parquet",
    },
    "human_playwright": {
        "label": "Human Playwright",
        "steps_path": CANONICAL_DIR / "human_playwright_steps.parquet",
        "trajectories_path": CANONICAL_DIR / "human_playwright_trajectories.parquet",
    },
    "vwa_gpt4v_som": {
        "label": "VWA GPT-4V + SoM",
        "steps_path": CANONICAL_DIR / "vwa_gpt4v_som_steps.parquet",
        "trajectories_path": CANONICAL_DIR / "vwa_gpt4v_som_trajectories.parquet",
    },
}

STEP_COLUMNS = [
    "sample_id",
    "source",
    "trajectory_id",
    "step_idx",
    "website",
    "domain",
    "subdomain",
    "goal",
    "actor_type",
    "action_api_name",
    "action_method",
    "action_selector",
    "action_value",
    "action_text",
    "action_raw_json",
    "obs_prev_image_ref",
    "obs_prev_image_path",
    "obs_next_image_ref",
    "obs_next_image_path",
    "is_terminal",
    "final_success",
    "needs_goal_mapping",
    "source_file",
    "source_action_uid",
]


def _safe_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value if value.strip() else None
    return str(value)


def _truncate(value: str | None, limit: int = 140) -> str | None:
    if not value:
        return value
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "…"


def available_canonical_sources() -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for key, info in CANONICAL_SOURCES.items():
        steps_path = info["steps_path"]
        traj_path = info["trajectories_path"]
        if not steps_path.exists() or not traj_path.exists():
            continue
        steps_rows = pq.ParquetFile(steps_path).metadata.num_rows
        traj_rows = pq.ParquetFile(traj_path).metadata.num_rows
        sources.append(
            {
                "key": key,
                "label": info["label"],
                "steps_path": steps_path,
                "trajectories_path": traj_path,
                "num_steps": steps_rows,
                "num_trajectories": traj_rows,
            }
        )
    return sources


def _format_goal(row: dict[str, Any]) -> str:
    goal = _safe_text(row.get("goal"))
    if goal:
        return goal
    if row.get("needs_goal_mapping"):
        return "（该轨迹当前没有任务目标映射，请优先根据页面变化与动作本身进行标注）"
    return "（无任务描述）"


def _format_action(row: dict[str, Any]) -> str:
    method = _safe_text(row.get("action_method")) or _safe_text(row.get("action_api_name")) or "action"
    args: list[str] = []

    selector = _truncate(_safe_text(row.get("action_selector")), 120)
    value = _truncate(_safe_text(row.get("action_value")), 80)
    action_text = _truncate(_safe_text(row.get("action_text")), 120)

    if selector:
        args.append(f"selector={selector!r}")
    if value:
        args.append(f"value={value!r}")
    if action_text and action_text != value:
        args.append(f"target={action_text!r}")

    if args:
        return f"{method}({', '.join(args)})"
    return method


def _format_action_desc(row: dict[str, Any]) -> str:
    parts = []
    source = _safe_text(row.get("source"))
    website = _safe_text(row.get("website"))
    domain = _safe_text(row.get("domain"))
    step_idx = row.get("step_idx")

    if source:
        parts.append(f"来源: {source}")
    if website:
        parts.append(f"站点: {website}")
    if domain:
        parts.append(f"域名: {domain}")
    if step_idx is not None:
        parts.append(f"Step: {step_idx}")

    raw_json = _safe_text(row.get("action_raw_json"))
    if raw_json:
        try:
            parsed = json.loads(raw_json)
            pretty = []
            for key in ["original_op", "op", "value"]:
                value = _safe_text(parsed.get(key))
                if value:
                    pretty.append(f"{key}={value}")
            if pretty:
                parts.append("动作元数据: " + ", ".join(pretty))
        except json.JSONDecodeError:
            pass

    return " | ".join(parts)


def _make_step_record(row: dict[str, Any]) -> dict[str, Any]:
    record = dict(row)
    record["step_id"] = row["sample_id"]
    record["task_id"] = row["trajectory_id"]
    record["task_goal"] = _format_goal(row)
    record["action"] = _format_action(row)
    record["action_desc"] = _format_action_desc(row)
    record["dataset_source"] = row["source"]
    record["source_label"] = CANONICAL_SOURCES.get(row["source"], {}).get("label", row["source"])
    record["screenshot_before"] = None
    record["screenshot_after"] = None
    return record


@lru_cache(maxsize=8)
def load_canonical_steps(selected_sources: tuple[str, ...]) -> list[dict[str, Any]]:
    normalized = tuple(sorted({source for source in selected_sources if source in CANONICAL_SOURCES}))
    if not normalized:
        return []

    frames = []
    for source in normalized:
        info = CANONICAL_SOURCES[source]
        steps_path = info["steps_path"]
        if not steps_path.exists():
            continue
        frame = pd.read_parquet(steps_path, columns=STEP_COLUMNS)
        frames.append(frame)

    if not frames:
        return []

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(["source", "trajectory_id", "step_idx"], kind="stable")
    return [_make_step_record(row) for row in merged.to_dict("records")]


@lru_cache(maxsize=1024)
def _load_human_image_bytes(source_file: str, resource_path: str) -> bytes | None:
    zip_path = ROOT / source_file
    if not zip_path.exists():
        return None
    try:
        with zipfile.ZipFile(zip_path) as zf:
            return zf.read(resource_path)
    except Exception:
        return None


def _coerce_screenshot_bytes(screenshot: Any) -> bytes | None:
    if screenshot is None:
        return None
    if isinstance(screenshot, dict):
        screenshot = screenshot.get("bytes")
    if isinstance(screenshot, bytearray):
        return bytes(screenshot)
    if isinstance(screenshot, bytes):
        return screenshot
    if hasattr(screenshot, "tobytes"):
        return screenshot.tobytes()
    return None


@lru_cache(maxsize=2048)
def _load_mind2web_image_bytes(source_file: str, action_uid: str) -> bytes | None:
    parquet_path = ROOT / source_file
    if not parquet_path.exists():
        return None

    try:
        frame = pd.read_parquet(
            parquet_path,
            columns=["action_uid", "screenshot"],
            filters=[("action_uid", "==", action_uid)],
        )
    except Exception:
        frame = pd.read_parquet(parquet_path, columns=["action_uid", "screenshot"])
        frame = frame[frame["action_uid"] == action_uid]

    if frame.empty:
        return None
    return _coerce_screenshot_bytes(frame.iloc[0]["screenshot"])


def _parse_mind2web_ref(image_ref: str | None) -> tuple[str, str] | tuple[None, None]:
    if not image_ref:
        return None, None
    parts = image_ref.split("::", 2)
    if len(parts) < 2:
        return None, None
    return parts[0], parts[1]


def _parse_vwa_ref(image_ref: str | None) -> tuple[str | None, int | None]:
    if not image_ref:
        return None, None
    member_name, _, page_idx = image_ref.rpartition("::")
    if not member_name or page_idx == "":
        return None, None
    try:
        return member_name, int(page_idx)
    except ValueError:
        return None, None


@lru_cache(maxsize=256)
def _load_vwa_render_page_images(source_file: str, member_name: str) -> tuple[bytes | None, ...]:
    tar_path = ROOT / source_file
    if not tar_path.exists():
        return tuple()

    try:
        with tarfile.open(tar_path, "r") as tf:
            handle = tf.extractfile(member_name)
            if handle is None:
                return tuple()
            html_text = handle.read().decode("utf-8", errors="ignore")
    except Exception:
        return tuple()

    pages = parse_render_pages(html_text)
    return tuple(decode_data_url_image(page.get("image_data_url")) for page in pages)


def load_step_image_bytes(step: dict[str, Any], which: str) -> bytes | None:
    if which not in {"before", "after"}:
        raise ValueError(f"Unsupported image side: {which}")

    if step.get("dataset_source") == "human_playwright":
        resource_path = step.get("obs_prev_image_path") if which == "before" else step.get("obs_next_image_path")
        source_file = step.get("source_file")
        if source_file and resource_path:
            return _load_human_image_bytes(source_file, resource_path)
        return None

    if step.get("dataset_source") == "mind2web":
        image_ref = step.get("obs_prev_image_ref") if which == "before" else step.get("obs_next_image_ref")
        source_file, action_uid = _parse_mind2web_ref(image_ref)
        if source_file and action_uid:
            return _load_mind2web_image_bytes(source_file, action_uid)
        return None

    if step.get("dataset_source") == "vwa_gpt4v_som":
        image_ref = step.get("obs_prev_image_ref") if which == "before" else step.get("obs_next_image_ref")
        member_name, page_idx = _parse_vwa_ref(image_ref)
        source_file = step.get("source_file")
        if source_file and member_name is not None and page_idx is not None:
            pages = _load_vwa_render_page_images(source_file, member_name)
            if 0 <= page_idx < len(pages):
                return pages[page_idx]
        return None

    return None


def load_step_image(step: dict[str, Any], which: str) -> Image.Image | None:
    data = load_step_image_bytes(step, which)
    if not data:
        return None
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return None
