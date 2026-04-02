#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vwa_render_parser import decode_data_url_image, parse_render_pages  # noqa: E402


DEFAULT_BENCHMARK = ROOT / "data" / "benchmark_tasks" / "webpqb_benchmark.json"
DEFAULT_TEXT_AUDIT = ROOT / "data" / "benchmark_tasks" / "webpqb_benchmark_text_audit.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "benchmark_tasks" / "webpqb_benchmark_visual_audit.jsonl"
DEFAULT_RENDER_TAR = ROOT / "data" / "_raw" / "gpt4v_som_910"

SYSTEM_PROMPT = """You are a visual benchmark curator for Web-PQB.

You will receive:
- a benchmark task definition
- one webpage screenshot taken from a related VisualWebArena trajectory
- optional text-audit hints from a text LLM

Your goal is to perform a visual re-audit.

Return strict JSON with keys:
- visual_dependence_reaudit: one of ["high", "medium", "low", "unavailable"]
- distraction_visible_reaudit: one of ["yes", "no", "unsure", "n/a"]
- recovery_visual_mislead: one of ["low", "medium", "high", "unsure", "n/a"]
- keep_or_drop_revision: one of ["keep", "drop", "revise", "no_change"]
- confidence: float in [0, 1]
- rationale_short: short explanation for humans
- cot_text: brief internal reasoning trace

Guidelines:
- Focus on whether the screenshot really shows meaningful visual ambiguity or distractors.
- For recovery tasks, judge whether the visible state seems likely to mislead an agent.
- If the screenshot is insufficient, use "unsure" or "unavailable".
- Do not rewrite the whole task; suggest revision only when visual evidence contradicts the current benchmark label.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run visual re-audit on benchmark tasks with available VWA screenshots.")
    parser.add_argument("--model-path", required=True, help="Local HF path or model id for a VLM.")
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK))
    parser.add_argument("--text-audit", default=str(DEFAULT_TEXT_AUDIT))
    parser.add_argument("--render-tar", default=str(DEFAULT_RENDER_TAR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    parser.add_argument(
        "--select-mode",
        default="flagged",
        choices=["all", "flagged", "recovery_only", "high_distraction_only"],
        help="Which tasks to visually re-audit.",
    )
    return parser.parse_args()


def load_json_list(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"Expected a JSON list in {path}")
    return data


def load_text_audit(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return result
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            bench_id = rec.get("benchmark_id")
            if bench_id:
                result[str(bench_id)] = rec
    return result


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
    first = text.find("{")
    last = text.rfind("}")
    if first == -1 or last == -1 or last <= first:
        raise ValueError("No JSON object found in model output.")
    return json.loads(text[first : last + 1])


def norm_choice(value: Any, allowed: set[str], default: str) -> str:
    if value is None:
        return default
    text = str(value).strip().lower()
    return text if text in allowed else default


def norm_confidence(value: Any) -> float | None:
    if value is None:
        return None
    try:
        score = float(value)
    except Exception:
        return None
    return max(0.0, min(1.0, score))


def task_render_member(task: dict[str, Any]) -> str | None:
    site = task.get("site")
    source_task_id = task.get("task_id")
    if task.get("category") == "recovery":
        source_task_id = task.get("source_task_id")
    if site is None or source_task_id in (None, ""):
        return None
    return f"gpt4v_som/{site}_gpt4v_som/render_{int(source_task_id)}.html"


@lru_cache(maxsize=2048)
def load_task_screenshot_bytes(render_tar: str, member_name: str) -> bytes | None:
    tar_path = Path(render_tar)
    if not tar_path.exists():
        return None
    try:
        with tarfile.open(tar_path, "r") as tf:
            handle = tf.extractfile(member_name)
            if handle is None:
                return None
            html_text = handle.read().decode("utf-8", errors="ignore")
    except Exception:
        return None

    pages = parse_render_pages(html_text)
    if not pages:
        return None
    return decode_data_url_image(pages[0].get("image_data_url"))


def load_task_image(render_tar: str, task: dict[str, Any]) -> Image.Image | None:
    member_name = task_render_member(task)
    if not member_name:
        return None
    payload = load_task_screenshot_bytes(render_tar, member_name)
    if not payload:
        return None
    try:
        return Image.open(io.BytesIO(payload)).convert("RGB")
    except Exception:
        return None


def should_select(task: dict[str, Any], text_rec: dict[str, Any], mode: str) -> bool:
    category = task.get("category", "")
    if mode == "all":
        return True
    if mode == "recovery_only":
        return category == "recovery"
    if mode == "high_distraction_only":
        return category == "high_distraction"
    if category in {"high_distraction", "recovery"}:
        return True
    if text_rec.get("keep_or_drop") in {"drop", "revise"}:
        return True
    if text_rec.get("category_correct") in {"no", "unsure"}:
        return True
    try:
        conf = float(text_rec.get("confidence"))
    except Exception:
        conf = None
    return conf is None or conf < 0.75


def build_user_prompt(task: dict[str, Any], text_rec: dict[str, Any], screenshot_available: bool) -> str:
    lines = [
        f"benchmark_id: {task.get('benchmark_id', '')}",
        f"category: {task.get('category', '')}",
        f"site: {task.get('site', '')}",
        f"intent: {task.get('intent', '')}",
        f"start_url: {task.get('start_url', '')}",
        f"wrong_start_url: {task.get('wrong_start_url', '')}",
        f"original_start_url: {task.get('original_start_url', '')}",
        f"screenshot_available: {screenshot_available}",
        "",
        "eval:",
        json.dumps(task.get("eval", {}), ensure_ascii=False, indent=2),
    ]
    if task.get("construction_note"):
        lines.extend(["", "construction_note:", str(task["construction_note"])])
    if text_rec:
        lines.extend(
            [
                "",
                "text_audit_hints:",
                json.dumps(
                    {
                        "keep_or_drop": text_rec.get("keep_or_drop"),
                        "category_correct": text_rec.get("category_correct"),
                        "task_solvable": text_rec.get("task_solvable"),
                        "evaluator_reliable": text_rec.get("evaluator_reliable"),
                        "visual_dependence": text_rec.get("visual_dependence"),
                        "confidence": text_rec.get("confidence"),
                        "rationale_short": text_rec.get("rationale_short"),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            ]
        )
    if task.get("category") == "recovery":
        lines.extend(
            [
                "",
                "Important note:",
                "For recovery tasks, the screenshot may come from the related source task trajectory, not the injected wrong_start_url itself.",
                "Use the screenshot only as supporting context, not as the sole basis for judging recovery validity.",
            ]
        )
    lines.append("")
    lines.append("Return JSON only.")
    return "\n".join(lines)


class VLMJudge:
    def __init__(self, args: argparse.Namespace):
        import torch
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor

        try:
            from transformers import AutoModelForImageTextToText as VisionModel
        except ImportError:
            try:
                from transformers import AutoModelForVision2Seq as VisionModel
            except ImportError:
                from transformers import Qwen2_5_VLForConditionalGeneration as VisionModel

        dtype_map = {
            "auto": "auto",
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.args = args
        self.process_vision_info = process_vision_info
        self.processor = AutoProcessor.from_pretrained(
            args.model_path,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
        )
        self.model = VisionModel.from_pretrained(
            args.model_path,
            torch_dtype=dtype_map[args.torch_dtype],
            device_map=args.device_map,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
        )

    def generate(self, task: dict[str, Any], text_rec: dict[str, Any], image: Image.Image | None) -> tuple[dict[str, Any], str]:
        content: list[dict[str, Any]] = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": build_user_prompt(task, text_rec, image is not None)})
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = self.process_vision_info(messages)
        model_inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        model_inputs = model_inputs.to(self.model.device)
        generated = self.model.generate(
            **model_inputs,
            max_new_tokens=self.args.max_new_tokens,
            do_sample=self.args.temperature > 0,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
        )
        trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated)
        ]
        raw_text = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        parsed = extract_json(raw_text)
        return parsed, raw_text


def load_existing_predictions(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            benchmark_id = rec.get("benchmark_id")
            if benchmark_id:
                done.add(str(benchmark_id))
    return done


def write_jsonl(path: Path, record: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    benchmark = load_json_list(Path(args.benchmark))
    text_audit = load_text_audit(Path(args.text_audit))

    selected = []
    for task in benchmark:
        text_rec = text_audit.get(str(task.get("benchmark_id")), {})
        if should_select(task, text_rec, args.select_mode):
            selected.append(task)
    if args.limit is not None:
        selected = selected[: args.limit]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = load_existing_predictions(output_path) if args.resume else set()

    judge = VLMJudge(args)
    model_name = Path(args.model_path).name.rstrip("/") or args.model_path
    total = len(selected)

    for idx, task in enumerate(selected, start=1):
        benchmark_id = str(task.get("benchmark_id"))
        if benchmark_id in done_ids:
            continue
        text_rec = text_audit.get(benchmark_id, {})
        image = load_task_image(args.render_tar, task)
        try:
            parsed, raw_text = judge.generate(task, text_rec, image)
            record = {
                "benchmark_id": benchmark_id,
                "category": task.get("category"),
                "site": task.get("site"),
                "task_id": task.get("task_id", task.get("source_task_id")),
                "model_name": model_name,
                "screenshot_available": image is not None,
                "parse_ok": True,
                "visual_dependence_reaudit": norm_choice(
                    parsed.get("visual_dependence_reaudit"),
                    {"high", "medium", "low", "unavailable"},
                    "unavailable" if image is None else "medium",
                ),
                "distraction_visible_reaudit": norm_choice(
                    parsed.get("distraction_visible_reaudit"),
                    {"yes", "no", "unsure", "n/a"},
                    "n/a",
                ),
                "recovery_visual_mislead": norm_choice(
                    parsed.get("recovery_visual_mislead"),
                    {"low", "medium", "high", "unsure", "n/a"},
                    "n/a",
                ),
                "keep_or_drop_revision": norm_choice(
                    parsed.get("keep_or_drop_revision"),
                    {"keep", "drop", "revise", "no_change"},
                    "no_change",
                ),
                "confidence": norm_confidence(parsed.get("confidence")),
                "rationale_short": str(parsed.get("rationale_short", "")),
                "cot_text": str(parsed.get("cot_text", "")),
                "raw_response": raw_text,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as exc:
            record = {
                "benchmark_id": benchmark_id,
                "category": task.get("category"),
                "site": task.get("site"),
                "task_id": task.get("task_id", task.get("source_task_id")),
                "model_name": model_name,
                "screenshot_available": image is not None,
                "parse_ok": False,
                "error": repr(exc),
                "timestamp": datetime.now().isoformat(),
            }
        write_jsonl(output_path, record)
        print(
            f"[{idx}/{total}] {benchmark_id} -> "
            f"{record.get('keep_or_drop_revision')} "
            f"visual={record.get('visual_dependence_reaudit')}"
        )


if __name__ == "__main__":
    main()
