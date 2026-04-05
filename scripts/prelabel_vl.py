#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from canonical_data import available_canonical_sources, load_canonical_steps, load_step_image, load_step_image_bytes  # noqa: E402


DEFAULT_SYSTEM_PROMPT = """You are a Web-PQB prelabeling judge.
Given a task goal, a before screenshot, an action, and an after screenshot:

1. Predict V_t in {-1, 0, +1}
2. Predict y_t in {0, 1}
3. Give a confidence score in [0, 1]
4. Provide a short rationale grounded in visible evidence
5. Provide a concise chain-of-thought style explanation in cot_text

Definitions:
- V_t = +1 if the action clearly advances the task
- V_t = 0 if the action is neutral / exploratory but not harmful
- V_t = -1 if the action moves away from the goal or causes a visible error
- y_t = 1 if the action is an optimal / high-quality main-path move or an effective recovery
- y_t = 0 if the action is suboptimal, a detour, or off-track

Return strict JSON only with keys:
pred_V_t, pred_y_t, confidence, rationale_short, cot_text
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use a local VLM to prelabel Web-PQB steps.")
    parser.add_argument(
        "--sources",
        nargs="+",
        required=True,
        help="Canonical sources to load, e.g. vwa_gpt4v_som mind2web",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Local HF snapshot path or model id, e.g. /root/autodl-tmp/.../Qwen2.5-VL-7B-Instruct",
    )
    parser.add_argument(
        "--remote-url",
        default=None,
        help="Optional remote inference endpoint, e.g. http://127.0.0.1:8008/predict",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "predictions" / "predictions_qwen25vl7b.jsonl"),
        help="Where to append prediction JSONL records.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of steps to process.")
    parser.add_argument("--start-index", type=int, default=0, help="Start offset after sorting.")
    parser.add_argument("--resume", action="store_true", help="Skip step_ids already present in output.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--html-chars", type=int, default=0, help="Reserved for future use.")
    parser.add_argument("--remote-timeout", type=int, default=180, help="HTTP timeout in seconds for remote inference.")
    args = parser.parse_args()
    if not args.model_path and not args.remote_url:
        parser.error("Either --model-path or --remote-url must be provided.")
    return args


def normalize_source_keys(source_keys: list[str]) -> tuple[str, ...]:
    available = {info["key"] for info in available_canonical_sources()}
    invalid = [source for source in source_keys if source not in available]
    if invalid:
        raise SystemExit(f"Unknown or unavailable canonical source(s): {invalid}")
    return tuple(source_keys)


def load_existing_predictions(path: Path) -> tuple[set[str], dict[tuple[str, int], float]]:
    done: set[str] = set()
    trajectory_state: dict[tuple[str, int], float] = {}
    if not path.exists():
        return done, trajectory_state

    last_by_step: dict[str, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            step_id = rec.get("step_id")
            if step_id:
                last_by_step[step_id] = rec

    for rec in last_by_step.values():
        step_id = rec.get("step_id")
        trajectory_id = rec.get("trajectory_id")
        step_idx = rec.get("step_idx")
        pred_o = rec.get("pred_O_t")
        if step_id:
            done.add(step_id)
        try:
            if trajectory_id is not None and step_idx is not None and pred_o is not None:
                trajectory_state[(str(trajectory_id), int(step_idx))] = float(pred_o)
        except Exception:
            continue

    return done, trajectory_state


def compute_prev_state_from_existing(steps: list[dict[str, Any]], existing_by_stepidx: dict[tuple[str, int], float]) -> dict[str, float]:
    current: dict[str, float] = {}
    for step in steps:
        trajectory_id = str(step.get("trajectory_id", step.get("task_id", "")))
        step_idx = int(step.get("step_idx", 0))
        if (trajectory_id, step_idx) in existing_by_stepidx:
            current[trajectory_id] = existing_by_stepidx[(trajectory_id, step_idx)]
    return current


def sanitize_text(text: str | None, limit: int = 600) -> str:
    if not text:
        return ""
    text = " ".join(str(text).split())
    return text[:limit]


def build_user_prompt(step: dict[str, Any]) -> str:
    goal = step.get("task_goal") or "（无任务目标）"
    action = step.get("action") or "（无动作）"
    action_desc = step.get("action_desc") or ""
    website = step.get("website") or ""
    domain = step.get("domain") or ""
    return (
        "You will see two screenshots from a web trajectory.\n"
        "The first image is the state BEFORE the action.\n"
        "The second image is the state AFTER the action.\n\n"
        f"Task goal:\n{goal}\n\n"
        f"Website: {website}\n"
        f"Domain: {domain}\n\n"
        f"Action:\n{action}\n\n"
        f"Action metadata:\n{sanitize_text(action_desc, 400)}\n\n"
        "Judge whether the action advanced the goal and whether it was a high-quality action.\n"
        "Use visible evidence from the screenshots and the action description.\n"
        "Return strict JSON only."
    )


def extract_json_object(text: str) -> dict[str, Any]:
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


def coerce_v(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip().replace("＋", "+")
    mapping = {"+1": 1, "1": 1, "0": 0, "-1": -1, "- 1": -1}
    if value in mapping:
        return mapping[value]
    try:
        num = int(value)
    except Exception:
        return None
    return num if num in {-1, 0, 1} else None


def coerce_y(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
    mapping = {"1": 1, "0": 0, True: 1, False: 0}
    if value in mapping:
        return mapping[value]
    try:
        num = int(value)
    except Exception:
        return None
    return num if num in {0, 1} else None


def coerce_confidence(value: Any) -> float | None:
    if value is None:
        return None
    try:
        score = float(value)
    except Exception:
        return None
    if score < 0:
        return 0.0
    if score > 1:
        return 1.0
    return score


def compute_current_o(prev_o: float | None, y_t: int | None, gamma: float = 0.5) -> float | None:
    if prev_o is None or y_t is None:
        return None
    return 1.0 if y_t == 1 else gamma * prev_o


def jsonl_append(path: Path, record: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


class QwenVLJudge:
    def __init__(self, args: argparse.Namespace):
        from transformers import AutoProcessor

        try:
            from transformers import AutoModelForImageTextToText as VisionModel
        except ImportError:
            try:
                from transformers import AutoModelForVision2Seq as VisionModel
            except ImportError:
                from transformers import Qwen2_5_VLForConditionalGeneration as VisionModel

        from qwen_vl_utils import process_vision_info

        import torch

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

    def predict(self, step: dict[str, Any]) -> tuple[dict[str, Any], str]:
        before = load_step_image(step, "before")
        after = load_step_image(step, "after")
        if before is None or after is None:
            raise RuntimeError("Missing before/after screenshot for this step.")

        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": before},
                    {"type": "image", "image": after},
                    {"type": "text", "text": build_user_prompt(step)},
                ],
            },
        ]

        prompt_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = self.process_vision_info(messages)
        model_inputs = self.processor(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        model_inputs = model_inputs.to(self.model.device)

        generate_kwargs = {
            "max_new_tokens": self.args.max_new_tokens,
            "do_sample": self.args.temperature > 0,
            "temperature": self.args.temperature,
            "top_p": self.args.top_p,
        }
        generated = self.model.generate(**model_inputs, **generate_kwargs)
        trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated)
        ]
        raw_text = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        parsed = extract_json_object(raw_text)
        return parsed, raw_text


class RemoteVLJudge:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.url = args.remote_url
        self.timeout = args.remote_timeout

    def predict(self, step: dict[str, Any]) -> tuple[dict[str, Any], str]:
        before = load_step_image_bytes(step, "before")
        after = load_step_image_bytes(step, "after")
        if before is None or after is None:
            raise RuntimeError("Missing before/after screenshot for this step.")

        payload = {
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
            "user_prompt": build_user_prompt(step),
            "before_image_b64": base64.b64encode(before).decode("ascii"),
            "after_image_b64": base64.b64encode(after).decode("ascii"),
            "generation": {
                "max_new_tokens": self.args.max_new_tokens,
                "temperature": self.args.temperature,
                "top_p": self.args.top_p,
            },
        }

        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Remote inference HTTP {exc.code}: {details}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Remote inference connection failed: {exc}") from exc

        try:
            result = json.loads(body)
        except Exception as exc:
            raise RuntimeError(f"Remote inference returned non-JSON response: {body[:300]}") from exc

        if "error" in result:
            raise RuntimeError(f"Remote inference error: {result['error']}")

        parsed = result.get("parsed")
        raw_text = result.get("raw_text", "")
        if not isinstance(parsed, dict):
            raise RuntimeError(f"Remote inference returned invalid payload: {result}")
        return parsed, raw_text


def main() -> None:
    args = parse_args()
    selected_sources = normalize_source_keys(args.sources)
    steps = load_canonical_steps(selected_sources)
    if args.start_index:
        steps = steps[args.start_index :]
    if args.limit is not None:
        steps = steps[: args.limit]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set[str] = set()
    existing_step_o: dict[tuple[str, int], float] = {}
    if args.resume:
        done_ids, existing_step_o = load_existing_predictions(output_path)

    trajectory_o_state = compute_prev_state_from_existing(steps, existing_step_o)
    judge = RemoteVLJudge(args) if args.remote_url else QwenVLJudge(args)

    total = len(steps)
    for offset, step in enumerate(steps, start=1):
        step_id = step["step_id"]
        if step_id in done_ids:
            continue

        trajectory_id = str(step.get("trajectory_id", step.get("task_id", "")))
        step_idx = int(step.get("step_idx", 0))
        prev_o = trajectory_o_state.get(trajectory_id, 1.0)

        record: dict[str, Any] = {
            "step_id": step_id,
            "task_id": step.get("task_id"),
            "source": step.get("dataset_source"),
            "trajectory_id": trajectory_id,
            "step_idx": step_idx,
            "model_name": Path(args.model_path).name.rstrip("/") if args.model_path else "remote_vl_judge",
            "prompt_version": "webpqb_v1",
            "gamma_penalty": 0.5,
            "pred_O_prev": prev_o,
            "timestamp": datetime.now().isoformat(),
        }

        try:
            parsed, raw_text = judge.predict(step)
            pred_v = coerce_v(parsed.get("pred_V_t", parsed.get("V_t")))
            pred_y = coerce_y(parsed.get("pred_y_t", parsed.get("y_t")))
            pred_conf = coerce_confidence(parsed.get("confidence"))
            pred_o = compute_current_o(prev_o, pred_y)
            pred_r = (pred_v * pred_o) if (pred_v is not None and pred_o is not None) else None

            record.update(
                {
                    "parse_ok": pred_v is not None and pred_y is not None,
                    "pred_V_t": pred_v,
                    "pred_y_t": pred_y,
                    "pred_confidence": pred_conf,
                    "rationale_short": parsed.get("rationale_short", ""),
                    "cot_text": parsed.get("cot_text", ""),
                    "pred_O_t": pred_o,
                    "pred_R_t": pred_r,
                    "raw_response": raw_text,
                }
            )
            if pred_o is not None:
                trajectory_o_state[trajectory_id] = pred_o
        except Exception as exc:
            record.update(
                {
                    "parse_ok": False,
                    "pred_V_t": None,
                    "pred_y_t": None,
                    "pred_confidence": None,
                    "rationale_short": "",
                    "cot_text": "",
                    "pred_O_t": None,
                    "pred_R_t": None,
                    "error": repr(exc),
                }
            )

        jsonl_append(output_path, record)
        print(
            f"[{offset}/{total}] {step_id} -> "
            f"V={record.get('pred_V_t')} y={record.get('pred_y_t')} "
            f"O={record.get('pred_O_t')}"
        )


if __name__ == "__main__":
    main()
