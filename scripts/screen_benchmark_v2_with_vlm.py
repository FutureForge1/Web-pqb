#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from benchmark_v2_common import BENCHMARK_V2_DIR, read_jsonl


DEFAULT_TASK_CARDS = BENCHMARK_V2_DIR / "task_cards_hard_filtered.jsonl"
DEFAULT_MANIFEST = BENCHMARK_V2_DIR / "page_capture_manifest.jsonl"
DEFAULT_OUTPUT = BENCHMARK_V2_DIR / "vlm_screening.jsonl"

SYSTEM_PROMPT = """You are a multimodal curator for Web-PQB-v2.

You are given:
- a benchmark task card
- one or more webpage screenshots captured from the true start page or recovery pages

Your job is to produce structured screening judgments. Be conservative.

Rules:
- This is a screening pass, not a final benchmark decision.
- Prefer "unsure" instead of over-claiming.
- Prefer "revise" over "drop" unless the evidence clearly shows the task is unusable.
- Recovery judgments must focus on whether the wrong start is visibly wrong-but-recoverable.
- For normal base tasks, judge whether the shown page is a reasonable benchmark starting state, not whether the final target item is already visible.
- Homepages, category landing pages, search pages, and listing pages can still support a task even if the exact target product/category is not yet on screen.
- Use "category_supported=no" only when the page is clearly broken, off-site, blank, irrelevant, or obviously incompatible with the task.
- For base tasks, all recovery_* fields must be "n/a".
- For high_distraction tasks, if the screenshot does not yet reveal enough clutter evidence, prefer "unsure" over "no".

Return JSON only.
Do not use markdown fences.
Do not output any text before or after the JSON object.
Your response must begin with "{" and end with "}".
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multimodal screening for Web-PQB v2 task cards.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--task-cards", default=str(DEFAULT_TASK_CARDS))
    parser.add_argument("--capture-manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=420)
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    return parser.parse_args()


def strip_think_blocks(text: str) -> str:
    while "<think>" in text and "</think>" in text:
        start = text.find("<think>")
        end = text.find("</think>", start)
        if end == -1:
            break
        text = text[:start] + text[end + len("</think>") :]
    return text.strip()


def extract_json(text: str) -> dict[str, Any]:
    text = strip_think_blocks(text.strip())
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
    text = str(value).strip().lower() if value is not None else ""
    return text if text in allowed else default


def norm_conf(value: Any) -> float | None:
    try:
        conf = float(value)
    except Exception:
        return None
    return max(0.0, min(1.0, conf))


def load_manifest(path: Path) -> dict[str, list[dict[str, Any]]]:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in read_jsonl(path):
        if row.get("status") != "ok":
            continue
        by_task.setdefault(str(row["task_card_id"]), []).append(row)
    return by_task


def active_primary_candidates(card: dict[str, Any]) -> list[str]:
    candidates = set(card.get("category_candidates", []))
    labels = []
    for label in ("multi_path", "high_distraction"):
        if label in candidates:
            labels.append(label)
    return labels


def build_user_prompt(card: dict[str, Any], capture_rows: list[dict[str, Any]]) -> str:
    capture_summary = [
        {
            "page_role": row.get("page_role"),
            "requested_url": row.get("requested_url"),
            "final_url": row.get("final_url"),
            "page_title": row.get("page_title"),
        }
        for row in capture_rows
    ]
    task_family = str(card.get("task_family", ""))
    primary_candidates = active_primary_candidates(card)
    family_guidance = [
        "Judgment target: decide whether the screenshots provide valid START-STATE evidence for the candidate benchmark category.",
        "Do not require the exact target product, category page, or final answer to already be visible on the screenshot.",
    ]
    if task_family == "recovery":
        family_guidance.extend(
            [
                "This is a recovery task.",
                "Focus on whether the wrong-start page is visibly wrong but still recoverable.",
                "Use recovery_* fields based on the visible mismatch, recoverability, misleadingness, and answer leakage.",
            ]
        )
    else:
        family_guidance.extend(
            [
                "This is a normal base task.",
                "Judge whether the shown page is a plausible and useful start page for the task on the correct site.",
                "For base tasks, set every recovery_* field to \"n/a\".",
            ]
        )
    if "multi_path" in primary_candidates:
        family_guidance.append(
            "multi_path=yes when the page offers multiple plausible navigation/search/filter routes; a homepage can still qualify."
        )
    if "high_distraction" in primary_candidates:
        family_guidance.append(
            "high_distraction=yes only when visible clutter/confusable choices are actually evident; otherwise use unsure."
        )
    lines = [
        f"task_card_id: {card.get('task_card_id', '')}",
        f"task_family: {card.get('task_family', '')}",
        f"site: {card.get('site', '')}",
        f"intent: {card.get('intent', '')}",
        f"start_url: {card.get('start_url', '')}",
        f"original_start_url: {card.get('original_start_url', '')}",
        f"wrong_start_url: {card.get('wrong_start_url', '')}",
        f"visual_dependence_prior: {card.get('visual_dependence_prior', '')}",
        f"category_candidates: {card.get('category_candidates', [])}",
        "",
        "judgment_guidance:",
        *[f"- {line}" for line in family_guidance],
        "",
        "capture_summary:",
        json.dumps(capture_summary, ensure_ascii=False, indent=2),
        "",
        "Return one JSON object only. Example format:",
        "{",
        '  "category_supported": "yes",',
        '  "visual_dependence": "high",',
        '  "multi_path_valid": "yes",',
        '  "route_plurality": "many",',
        '  "distraction_visible": "n/a",',
        '  "distractor_density": "n/a",',
        '  "target_confusability": "n/a",',
        '  "recovery_wrong_start_valid": "n/a",',
        '  "recovery_recoverable": "n/a",',
        '  "recovery_misleadingness": "n/a",',
        '  "recovery_answer_leakage": "n/a",',
        '  "recovery_severity": "n/a",',
        '  "confidence": 0.81,',
        '  "rationale_short": "Short explanation.",',
        '  "cot_text": "Brief reasoning trace."',
        "}",
    ]
    return "\n".join(lines)


def derive_triage(card: dict[str, Any], parsed: dict[str, Any]) -> str:
    family = card.get("task_family")
    candidates = set(card.get("category_candidates", []))
    category_supported = parsed.get("category_supported", "unsure")

    if family == "recovery":
        if (
            parsed.get("recovery_wrong_start_valid") == "yes"
            and parsed.get("recovery_recoverable") == "yes"
            and parsed.get("recovery_answer_leakage") == "no"
            and parsed.get("recovery_misleadingness") in {"medium", "high"}
        ):
            return "keep"
        if parsed.get("recovery_answer_leakage") == "yes" or parsed.get("recovery_recoverable") == "no":
            return "drop"
        return "revise"

    keep_multi_path = "multi_path" in candidates and category_supported in {"yes", "unsure"} and parsed.get("multi_path_valid") == "yes"
    keep_distraction = (
        "high_distraction" in candidates
        and category_supported in {"yes", "unsure"}
        and parsed.get("distraction_visible") == "yes"
        and parsed.get("distractor_density") in {"medium", "high"}
    )
    if keep_multi_path or keep_distraction:
        return "keep"

    if "multi_path" in candidates:
        if category_supported == "no" and parsed.get("multi_path_valid") == "no":
            return "revise"
        return "revise"

    if "high_distraction" in candidates:
        if category_supported == "no" and parsed.get("distraction_visible") == "no":
            return "revise"
        return "revise"

    return "revise"


def sanitize_for_family(card: dict[str, Any], parsed: dict[str, Any]) -> dict[str, Any]:
    clean = dict(parsed)
    candidates = set(card.get("category_candidates", []))
    family = card.get("task_family")

    if family != "recovery":
        for key in [
            "recovery_wrong_start_valid",
            "recovery_recoverable",
            "recovery_misleadingness",
            "recovery_answer_leakage",
            "recovery_severity",
        ]:
            clean[key] = "n/a"

    if "multi_path" not in candidates:
        for key in ["multi_path_valid", "route_plurality"]:
            clean[key] = "n/a"

    if "high_distraction" not in candidates:
        for key in ["distraction_visible", "distractor_density", "target_confusability"]:
            clean[key] = "n/a"

    return clean


def load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row["task_card_id"]) for row in read_jsonl(path) if row.get("task_card_id")}


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

    def generate_raw(self, card: dict[str, Any], capture_rows: list[dict[str, Any]]) -> str:
        content: list[dict[str, Any]] = []
        for row in sorted(capture_rows, key=lambda r: str(r.get("page_role"))):
            screenshot_path = row.get("screenshot_path")
            if screenshot_path and Path(screenshot_path).exists():
                content.append({"type": "image", "image": Image.open(screenshot_path).convert("RGB")})
        content.append({"type": "text", "text": build_user_prompt(card, capture_rows)})
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)
        model_inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
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
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    cards = read_jsonl(Path(args.task_cards))
    if args.limit is not None:
        cards = cards[: args.limit]
    manifest = load_manifest(Path(args.capture_manifest))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = load_done_ids(output_path) if args.resume else set()

    judge = VLMJudge(args)
    model_name = Path(args.model_path).name.rstrip("/") or args.model_path

    total = len(cards)
    for idx, card in enumerate(cards, start=1):
        task_id = str(card["task_card_id"])
        if task_id in done_ids:
            continue
        capture_rows = manifest.get(task_id, [])
        raw_text = None
        try:
            raw_text = judge.generate_raw(card, capture_rows)
            parsed = sanitize_for_family(card, extract_json(raw_text))
            record = {
                "task_card_id": task_id,
                "base_task_card_id": card.get("base_task_card_id"),
                "site": card.get("site"),
                "task_family": card.get("task_family"),
                "category_candidates": card.get("category_candidates", []),
                "model_name": model_name,
                "parse_ok": True,
                "num_images": len(capture_rows),
                "category_supported": norm_choice(parsed.get("category_supported"), {"yes", "no", "unsure"}, "unsure"),
                "visual_dependence": norm_choice(parsed.get("visual_dependence"), {"high", "medium", "low", "unsure"}, "unsure"),
                "multi_path_valid": norm_choice(parsed.get("multi_path_valid"), {"yes", "no", "unsure", "n/a"}, "n/a"),
                "route_plurality": norm_choice(parsed.get("route_plurality"), {"single", "few", "many", "unsure", "n/a"}, "n/a"),
                "distraction_visible": norm_choice(parsed.get("distraction_visible"), {"yes", "no", "unsure", "n/a"}, "n/a"),
                "distractor_density": norm_choice(parsed.get("distractor_density"), {"low", "medium", "high", "unsure", "n/a"}, "n/a"),
                "target_confusability": norm_choice(parsed.get("target_confusability"), {"low", "medium", "high", "unsure", "n/a"}, "n/a"),
                "recovery_wrong_start_valid": norm_choice(parsed.get("recovery_wrong_start_valid"), {"yes", "no", "unsure", "n/a"}, "n/a"),
                "recovery_recoverable": norm_choice(parsed.get("recovery_recoverable"), {"yes", "no", "unsure", "n/a"}, "n/a"),
                "recovery_misleadingness": norm_choice(parsed.get("recovery_misleadingness"), {"low", "medium", "high", "unsure", "n/a"}, "n/a"),
                "recovery_answer_leakage": norm_choice(parsed.get("recovery_answer_leakage"), {"yes", "no", "unsure", "n/a"}, "n/a"),
                "recovery_severity": norm_choice(parsed.get("recovery_severity"), {"low", "medium", "high", "unsure", "n/a"}, "n/a"),
                "confidence": norm_conf(parsed.get("confidence")),
                "rationale_short": str(parsed.get("rationale_short", "")),
                "cot_text": str(parsed.get("cot_text", "")),
                "triage_label": derive_triage(card, parsed),
                "raw_response": raw_text,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as exc:
            record = {
                "task_card_id": task_id,
                "site": card.get("site"),
                "task_family": card.get("task_family"),
                "category_candidates": card.get("category_candidates", []),
                "model_name": model_name,
                "parse_ok": False,
                "num_images": len(capture_rows),
                "error": repr(exc),
                "raw_response": raw_text,
                "timestamp": datetime.now().isoformat(),
            }
        append_jsonl(output_path, record)
        print(f"[{idx}/{total}] {task_id} -> {record.get('triage_label')} parse={record.get('parse_ok')}")

    print(f"Screening output -> {output_path}")


if __name__ == "__main__":
    main()
