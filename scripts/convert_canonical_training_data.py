#!/usr/bin/env python3

import argparse
import json
import sys
import tarfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vwa_render_parser import (
    domain_from_url,
    parse_action_text,
    parse_header_metadata,
    parse_member_identity,
    parse_render_pages,
    parse_results_lookup,
)


VWA_DIR = ROOT / "data" / "vwa"
VWA_GPT4V_SOM_TAR = ROOT / "data" / "_raw" / "gpt4v_som_910"


STEP_SCHEMA = pa.schema(
    [
        ("sample_id", pa.string()),
        ("source", pa.string()),
        ("trajectory_id", pa.string()),
        ("step_idx", pa.int32()),
        ("website", pa.string()),
        ("domain", pa.string()),
        ("subdomain", pa.string()),
        ("goal", pa.large_string()),
        ("actor_type", pa.string()),
        ("action_api_name", pa.string()),
        ("action_method", pa.string()),
        ("action_selector", pa.large_string()),
        ("action_value", pa.large_string()),
        ("action_text", pa.large_string()),
        ("action_candidates_json", pa.large_string()),
        ("action_raw_json", pa.large_string()),
        ("obs_prev_html", pa.large_string()),
        ("obs_prev_html_format", pa.string()),
        ("obs_prev_image_ref", pa.large_string()),
        ("obs_prev_image_path", pa.large_string()),
        ("obs_prev_time", pa.float64()),
        ("obs_next_html", pa.large_string()),
        ("obs_next_html_format", pa.string()),
        ("obs_next_image_ref", pa.large_string()),
        ("obs_next_image_path", pa.large_string()),
        ("obs_next_time", pa.float64()),
        ("is_terminal", pa.bool_()),
        ("final_success", pa.bool_()),
        ("needs_goal_mapping", pa.bool_()),
        ("source_file", pa.large_string()),
        ("source_action_uid", pa.string()),
    ]
)


TRAJECTORY_SCHEMA = pa.schema(
    [
        ("trajectory_id", pa.string()),
        ("source", pa.string()),
        ("source_file", pa.large_string()),
        ("website", pa.string()),
        ("domain", pa.string()),
        ("subdomain", pa.string()),
        ("goal", pa.large_string()),
        ("actor_type", pa.string()),
        ("num_source_rows", pa.int32()),
        ("num_transitions", pa.int32()),
        ("final_success", pa.bool_()),
        ("needs_goal_mapping", pa.bool_()),
    ]
)


class StreamingParquetWriter:
    def __init__(self, out_path: Path, schema: pa.Schema):
        self.out_path = out_path
        self.tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        self.schema = schema
        self.writer: pq.ParquetWriter | None = None
        self.rows_written = 0
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        if self.tmp_path.exists():
            self.tmp_path.unlink()

    def write_rows(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        table = pa.Table.from_pylist(rows, schema=self.schema)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.tmp_path, self.schema, compression="zstd")
        self.writer.write_table(table)
        self.rows_written += len(rows)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
            self.tmp_path.replace(self.out_path)

    def abort(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        if self.tmp_path.exists():
            self.tmp_path.unlink()


def relpath_str(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def json_dumps(obj: Any) -> str | None:
    if obj is None:
        return None
    return json.dumps(normalize_json_value(obj), ensure_ascii=False, separators=(",", ":"))


def normalize_json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): normalize_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [normalize_json_value(v) for v in value]
    if hasattr(value, "tolist"):
        return normalize_json_value(value.tolist())
    return str(value)


def first_non_null(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            if value == "":
                continue
            return value
        return str(value)
    return None


def snapshot_html(snapshot: dict[str, Any] | None) -> str | None:
    if not snapshot:
        return None
    return json_dumps(snapshot.get("html"))


def choose_before_frame(frames: list[dict[str, Any]], ts: float) -> dict[str, Any] | None:
    before = [frame for frame in frames if frame.get("timestamp", -1) <= ts]
    if before:
        return before[-1]
    return frames[0] if frames else None


def choose_after_frame(frames: list[dict[str, Any]], ts: float) -> dict[str, Any] | None:
    after = [frame for frame in frames if frame.get("timestamp", -1) >= ts]
    if after:
        return after[0]
    return frames[-1] if frames else None


def normalize_mind2web_operation(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}


def load_vwa_intent_lookup(vwa_dir: Path) -> dict[tuple[str, str], str]:
    lookup: dict[tuple[str, str], str] = {}
    site_dirs = {
        "classifieds": vwa_dir / "test_classifieds",
        "reddit": vwa_dir / "test_reddit",
        "shopping": vwa_dir / "test_shopping",
    }

    for site, site_dir in site_dirs.items():
        if not site_dir.exists():
            continue
        for json_path in sorted(site_dir.glob("*.json")):
            try:
                payload = json.loads(json_path.read_text())
            except Exception:
                continue
            task_id = first_non_null(payload.get("task_id"), json_path.stem)
            intent = first_non_null(payload.get("intent"))
            if task_id and intent:
                lookup[(site, task_id)] = intent

    return lookup


def iter_mind2web_rows(parquet_path: Path) -> Iterable[tuple[dict[str, Any], list[dict[str, Any]]]]:
    df = pd.read_parquet(
        parquet_path,
        columns=[
            "annotation_id",
            "website",
            "domain",
            "subdomain",
            "action_uid",
            "operation",
            "confirmed_task",
            "cleaned_html",
            "screenshot",
            "target_action_reprs",
            "action_reprs",
        ],
    )

    for annotation_id, group in df.groupby("annotation_id", sort=False):
        group = group.reset_index(drop=True)
        if len(group) < 2:
            continue

        trajectory_id = f"mind2web:{annotation_id}"
        source_file = relpath_str(parquet_path)
        website = first_non_null(group.loc[0, "website"])
        domain = first_non_null(group.loc[0, "domain"])
        subdomain = first_non_null(group.loc[0, "subdomain"])
        goal = first_non_null(group.loc[0, "confirmed_task"])

        trajectory = {
            "trajectory_id": trajectory_id,
            "source": "mind2web",
            "source_file": source_file,
            "website": website,
            "domain": domain,
            "subdomain": subdomain,
            "goal": goal,
            "actor_type": "human_demonstration",
            "num_source_rows": int(len(group)),
            "num_transitions": int(len(group) - 1),
            "final_success": True,
            "needs_goal_mapping": False,
        }

        rows: list[dict[str, Any]] = []
        for idx in range(len(group) - 1):
            prev = group.iloc[idx]
            nxt = group.iloc[idx + 1]
            op = normalize_mind2web_operation(prev["operation"])
            prev_shot = prev["screenshot"] or {}
            next_shot = nxt["screenshot"] or {}
            rows.append(
                {
                    "sample_id": f"{trajectory_id}:{idx}",
                    "source": "mind2web",
                    "trajectory_id": trajectory_id,
                    "step_idx": idx,
                    "website": website,
                    "domain": domain,
                    "subdomain": subdomain,
                    "goal": goal,
                    "actor_type": "human_demonstration",
                    "action_api_name": first_non_null(op.get("original_op")),
                    "action_method": first_non_null(op.get("op")),
                    "action_selector": None,
                    "action_value": first_non_null(op.get("value")),
                    "action_text": first_non_null(prev["target_action_reprs"]),
                    "action_candidates_json": json_dumps(prev["action_reprs"]),
                    "action_raw_json": json_dumps(op),
                    "obs_prev_html": first_non_null(prev["cleaned_html"]),
                    "obs_prev_html_format": "cleaned_html",
                    "obs_prev_image_ref": f"{source_file}::{prev['action_uid']}::{prev_shot.get('path', '')}",
                    "obs_prev_image_path": first_non_null(prev_shot.get("path")),
                    "obs_prev_time": None,
                    "obs_next_html": first_non_null(nxt["cleaned_html"]),
                    "obs_next_html_format": "cleaned_html",
                    "obs_next_image_ref": f"{source_file}::{nxt['action_uid']}::{next_shot.get('path', '')}",
                    "obs_next_image_path": first_non_null(next_shot.get("path")),
                    "obs_next_time": None,
                    "is_terminal": idx == len(group) - 2,
                    "final_success": True,
                    "needs_goal_mapping": False,
                    "source_file": source_file,
                    "source_action_uid": first_non_null(prev["action_uid"]),
                }
            )

        yield trajectory, rows


def convert_mind2web(mind2web_dir: Path, out_dir: Path) -> tuple[int, int]:
    steps_writer = StreamingParquetWriter(out_dir / "mind2web_steps.parquet", STEP_SCHEMA)
    traj_writer = StreamingParquetWriter(out_dir / "mind2web_trajectories.parquet", TRAJECTORY_SCHEMA)
    parquet_files = sorted(mind2web_dir.glob("*.parquet"))
    traj_count = 0

    try:
        for parquet_path in parquet_files:
            trajectory_rows: list[dict[str, Any]] = []
            step_rows: list[dict[str, Any]] = []
            for trajectory, rows in iter_mind2web_rows(parquet_path):
                trajectory_rows.append(trajectory)
                step_rows.extend(rows)
                traj_count += 1
            traj_writer.write_rows(trajectory_rows)
            steps_writer.write_rows(step_rows)
    except Exception:
        steps_writer.abort()
        traj_writer.abort()
        raise
    finally:
        pass

    steps_writer.close()
    traj_writer.close()

    return traj_count, steps_writer.rows_written


def parse_trace_file(
    trace_zip: Path,
    goal_lookup: dict[tuple[str, str], str] | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    with zipfile.ZipFile(trace_zip) as zf:
        raw = zf.read("trace.trace").decode("utf-8", errors="ignore").splitlines()

    before_events: list[dict[str, Any]] = []
    after_by_call: dict[str, dict[str, Any]] = {}
    frames_by_page: dict[str, list[dict[str, Any]]] = defaultdict(list)
    snapshots_by_name: dict[str, dict[str, Any]] = {}

    for line in raw:
        obj = json.loads(line)
        obj_type = obj.get("type")
        if obj_type == "before":
            before_events.append(obj)
        elif obj_type == "after":
            after_by_call[obj["callId"]] = obj
        elif obj_type == "screencast-frame":
            frames_by_page[obj["pageId"]].append(obj)
        elif obj_type == "frame-snapshot":
            snap = obj.get("snapshot", {})
            snapshots_by_name[snap.get("snapshotName")] = snap

    for page_frames in frames_by_page.values():
        page_frames.sort(key=lambda x: x.get("timestamp", -1))

    trace_rel = relpath_str(trace_zip)
    site = trace_zip.parent.name
    trace_name = trace_zip.name.replace(".trace.zip", "")
    trajectory_id = f"human_playwright:{site}:{trace_name}"
    goal = None
    if goal_lookup is not None:
        goal = goal_lookup.get((site, trace_name))

    valid_events = [
        event
        for event in before_events
        if event.get("callId") in after_by_call
    ]
    valid_events.sort(key=lambda x: x.get("startTime", -1))

    if not valid_events:
        return None, []

    first_snapshot = snapshots_by_name.get(valid_events[0].get("beforeSnapshot"))
    first_url = first_snapshot.get("frameUrl") if first_snapshot else None

    trajectory = {
        "trajectory_id": trajectory_id,
        "source": "human_playwright",
        "source_file": trace_rel,
        "website": site,
        "domain": domain_from_url(first_url),
        "subdomain": None,
        "goal": goal,
        "actor_type": "human_playwright",
        "num_source_rows": len(raw),
        "num_transitions": len(valid_events),
        "final_success": None,
        "needs_goal_mapping": goal is None,
    }

    rows: list[dict[str, Any]] = []
    for idx, before in enumerate(valid_events):
        after = after_by_call[before["callId"]]
        before_snap = snapshots_by_name.get(before.get("beforeSnapshot"))
        after_snap = snapshots_by_name.get(after.get("afterSnapshot"))
        page_id = before.get("pageId")
        page_frames = frames_by_page.get(page_id, [])
        before_frame = choose_before_frame(page_frames, before.get("startTime", -1))
        after_frame = choose_after_frame(page_frames, after.get("endTime", -1))
        params = before.get("params", {})

        rows.append(
            {
                "sample_id": f"{trajectory_id}:{idx}",
                "source": "human_playwright",
                "trajectory_id": trajectory_id,
                "step_idx": idx,
                "website": site,
                "domain": domain_from_url(first_url),
                "subdomain": None,
                "goal": goal,
                "actor_type": "human_playwright",
                "action_api_name": first_non_null(before.get("apiName")),
                "action_method": first_non_null(before.get("method")),
                "action_selector": first_non_null(params.get("selector")),
                "action_value": first_non_null(params.get("value"), params.get("text"), params.get("key")),
                "action_text": f"{before.get('apiName')}::{json_dumps(params)}",
                "action_candidates_json": None,
                "action_raw_json": json_dumps(params),
                "obs_prev_html": snapshot_html(before_snap),
                "obs_prev_html_format": "playwright_frame_snapshot_json",
                "obs_prev_image_ref": (
                    f"{trace_rel}::resources/{before_frame['sha1']}" if before_frame else None
                ),
                "obs_prev_image_path": (
                    f"resources/{before_frame['sha1']}" if before_frame else None
                ),
                "obs_prev_time": float(before.get("startTime")) if before.get("startTime") is not None else None,
                "obs_next_html": snapshot_html(after_snap),
                "obs_next_html_format": "playwright_frame_snapshot_json",
                "obs_next_image_ref": (
                    f"{trace_rel}::resources/{after_frame['sha1']}" if after_frame else None
                ),
                "obs_next_image_path": (
                    f"resources/{after_frame['sha1']}" if after_frame else None
                ),
                "obs_next_time": float(after.get("endTime")) if after.get("endTime") is not None else None,
                "is_terminal": idx == len(valid_events) - 1,
                "final_success": None,
                "needs_goal_mapping": goal is None,
                "source_file": trace_rel,
                "source_action_uid": first_non_null(before.get("callId")),
            }
        )

    return trajectory, rows


def convert_human_playwright(
    human_dir: Path,
    out_dir: Path,
    goal_lookup: dict[tuple[str, str], str] | None = None,
) -> tuple[int, int]:
    steps_writer = StreamingParquetWriter(out_dir / "human_playwright_steps.parquet", STEP_SCHEMA)
    traj_writer = StreamingParquetWriter(out_dir / "human_playwright_trajectories.parquet", TRAJECTORY_SCHEMA)
    trace_files = sorted(
        path
        for path in human_dir.glob("*/*.trace.zip")
        if path.parent.name != "_test"
    )
    traj_count = 0

    try:
        for trace_file in trace_files:
            trajectory, rows = parse_trace_file(trace_file, goal_lookup=goal_lookup)
            if trajectory is None:
                continue
            traj_writer.write_rows([trajectory])
            steps_writer.write_rows(rows)
            traj_count += 1
    except Exception:
        steps_writer.abort()
        traj_writer.abort()
        raise
    finally:
        pass

    steps_writer.close()
    traj_writer.close()

    return traj_count, steps_writer.rows_written


def load_vwa_results_from_tar(vwa_tar_path: Path) -> dict[tuple[str, str], bool]:
    lookup: dict[tuple[str, str], bool] = {}
    with tarfile.open(vwa_tar_path, "r") as tf:
        member = tf.next()
        while member is not None:
            if not (
                member.isfile()
                and member.name.endswith("results.txt")
                and "/._" not in member.name
            ):
                member = tf.next()
                continue
            handle = tf.extractfile(member)
            if handle is None:
                member = tf.next()
                continue
            text = handle.read().decode("utf-8", errors="ignore")
            lookup.update(parse_results_lookup(text))
            member = tf.next()
    return lookup


def iter_vwa_gpt4v_som_rows(
    vwa_tar_path: Path,
) -> Iterable[tuple[dict[str, Any], list[dict[str, Any]]]]:
    results_lookup = load_vwa_results_from_tar(vwa_tar_path)

    with tarfile.open(vwa_tar_path, "r") as tf:
        member = tf.next()
        while member is not None:
            if not (
                member.isfile()
                and member.name.endswith(".html")
                and "render_" in member.name
                and "/._" not in member.name
            ):
                member = tf.next()
                continue

            site, task_id = parse_member_identity(member.name)
            if not site or task_id is None:
                member = tf.next()
                continue

            handle = tf.extractfile(member)
            if handle is None:
                member = tf.next()
                continue
            html_text = handle.read().decode("utf-8", errors="ignore")
            metadata = parse_header_metadata(html_text)
            pages = parse_render_pages(html_text)
            if len(pages) < 2:
                member = tf.next()
                continue

            goal = first_non_null(metadata.get("intent"))
            start_url = first_non_null(metadata.get("start_url"), pages[0].get("url"))
            trajectory_id = f"vwa_gpt4v_som:{site}:{task_id}"
            source_file = relpath_str(vwa_tar_path)
            final_success = results_lookup.get((site, task_id))

            trajectory = {
                "trajectory_id": trajectory_id,
                "source": "vwa_gpt4v_som",
                "source_file": source_file,
                "website": site,
                "domain": domain_from_url(start_url),
                "subdomain": None,
                "goal": goal,
                "actor_type": "agent_gpt4v_som",
                "num_source_rows": len(pages),
                "num_transitions": len(pages) - 1,
                "final_success": final_success,
                "needs_goal_mapping": goal is None,
            }

            rows: list[dict[str, Any]] = []
            for idx in range(len(pages) - 1):
                current_page = pages[idx]
                next_page = pages[idx + 1]
                action = parse_action_text(current_page.get("parsed_action"))
                rows.append(
                    {
                        "sample_id": f"{trajectory_id}:{idx}",
                        "source": "vwa_gpt4v_som",
                        "trajectory_id": trajectory_id,
                        "step_idx": idx,
                        "website": site,
                        "domain": domain_from_url(current_page.get("url"))
                        or domain_from_url(start_url),
                        "subdomain": None,
                        "goal": goal,
                        "actor_type": "agent_gpt4v_som",
                        "action_api_name": action["api_name"],
                        "action_method": action["method"],
                        "action_selector": action["selector"],
                        "action_value": action["value"],
                        "action_text": action["text"],
                        "action_candidates_json": None,
                        "action_raw_json": json_dumps(
                            {
                                "prev_action": current_page.get("prev_action"),
                                "raw_prediction": current_page.get("raw_prediction"),
                                "parsed_action": current_page.get("parsed_action"),
                                "current_url": current_page.get("url"),
                                "next_url": next_page.get("url"),
                            }
                        ),
                        "obs_prev_html": first_non_null(current_page.get("state_obv")),
                        "obs_prev_html_format": "vwa_state_obv_text",
                        "obs_prev_image_ref": f"{member.name}::{idx}",
                        "obs_prev_image_path": f"{member.name}#page={idx}",
                        "obs_prev_time": None,
                        "obs_next_html": first_non_null(next_page.get("state_obv")),
                        "obs_next_html_format": "vwa_state_obv_text",
                        "obs_next_image_ref": f"{member.name}::{idx + 1}",
                        "obs_next_image_path": f"{member.name}#page={idx + 1}",
                        "obs_next_time": None,
                        "is_terminal": idx == len(pages) - 2,
                        "final_success": final_success,
                        "needs_goal_mapping": goal is None,
                        "source_file": source_file,
                        "source_action_uid": f"{member.name}:{idx}",
                    }
                )

            yield trajectory, rows
            member = tf.next()


def convert_vwa_gpt4v_som(vwa_tar_path: Path, out_dir: Path) -> tuple[int, int]:
    steps_writer = StreamingParquetWriter(out_dir / "vwa_gpt4v_som_steps.parquet", STEP_SCHEMA)
    traj_writer = StreamingParquetWriter(out_dir / "vwa_gpt4v_som_trajectories.parquet", TRAJECTORY_SCHEMA)
    traj_count = 0

    try:
        for trajectory, rows in iter_vwa_gpt4v_som_rows(vwa_tar_path):
            traj_writer.write_rows([trajectory])
            steps_writer.write_rows(rows)
            traj_count += 1
            if traj_count % 100 == 0:
                print(f"  processed {traj_count} trajectories...", flush=True)
    except Exception:
        steps_writer.abort()
        traj_writer.abort()
        raise

    steps_writer.close()
    traj_writer.close()
    return traj_count, steps_writer.rows_written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mind2web-dir",
        type=Path,
        default=ROOT / "data" / "multimodal_mind2web" / "data",
    )
    parser.add_argument(
        "--human-dir",
        type=Path,
        default=ROOT / "data" / "human_playwright_233",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "canonical",
    )
    parser.add_argument(
        "--vwa-gpt4v-som-tar",
        type=Path,
        default=VWA_GPT4V_SOM_TAR,
        help="Tar archive containing VisualWebArena GPT-4V+SoM render HTML trajectories.",
    )
    parser.add_argument(
        "--vwa-dir",
        type=Path,
        default=VWA_DIR,
        help="Optional VisualWebArena task directory used to recover human trajectory goals.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["mind2web", "human", "vwa_gpt4v_som"],
        default=["mind2web", "human", "vwa_gpt4v_som"],
        help="Only run the selected source converters.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if "mind2web" in args.sources:
        print(f"Converting Mind2Web from {args.mind2web_dir}")
        mind2web_traj, mind2web_steps = convert_mind2web(args.mind2web_dir, args.out_dir)
        print(f"Mind2Web trajectories={mind2web_traj} steps={mind2web_steps}")

    if "human" in args.sources:
        print(f"Converting Human Playwright from {args.human_dir}")
        goal_lookup = load_vwa_intent_lookup(args.vwa_dir) if args.vwa_dir.exists() else None
        if goal_lookup is None:
            print("VWA goal mapping not found; human goals will remain empty.")
        else:
            print(f"Loaded {len(goal_lookup)} VWA task goals from {args.vwa_dir}")
        human_traj, human_steps = convert_human_playwright(
            args.human_dir,
            args.out_dir,
            goal_lookup=goal_lookup,
        )
        print(f"Human trajectories={human_traj} steps={human_steps}")

    if "vwa_gpt4v_som" in args.sources:
        print(f"Converting VisualWebArena GPT-4V+SoM from {args.vwa_gpt4v_som_tar}")
        vwa_traj, vwa_steps = convert_vwa_gpt4v_som(args.vwa_gpt4v_som_tar, args.out_dir)
        print(f"VWA GPT-4V+SoM trajectories={vwa_traj} steps={vwa_steps}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
