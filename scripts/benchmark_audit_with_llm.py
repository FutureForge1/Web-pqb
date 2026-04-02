#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_INPUT = ROOT / "data" / "benchmark_tasks" / "webpqb_benchmark.json"
DEFAULT_OUTPUT = ROOT / "data" / "benchmark_tasks" / "webpqb_benchmark_text_audit.jsonl"

SYSTEM_PROMPT = """You are an expert benchmark curator for Web-PQB.

Your job is to review web-agent benchmark tasks and judge whether they belong in a high-quality process-quality benchmark.

You will receive:
- task category
- website
- task instruction
- start_url
- optional wrong_start_url / original_start_url
- evaluator metadata

Return strict JSON with these keys:
- keep_or_drop: one of ["keep", "drop", "revise"]
- category_correct: one of ["yes", "no", "unsure"]
- multi_path_confirmed: one of ["yes", "no", "unsure", "n/a"]
- distraction_visible: one of ["yes", "no", "unsure", "n/a"]
- recovery_setup_valid: one of ["yes", "no", "unsure", "n/a"]
- recovery_severity: one of ["low", "medium", "high", "n/a"]
- task_solvable: one of ["yes", "no", "unsure"]
- evaluator_reliable: one of ["yes", "no", "unsure"]
- duplicate_or_near_duplicate: one of ["yes", "no", "unsure"]
- visual_dependence: one of ["high", "medium", "low"]
- site_balance_priority: one of ["high", "medium", "low"]
- confidence: float in [0, 1]
- rationale_short: short explanation for human reviewers
- cot_text: brief internal reasoning trace

Guidelines:
- multi_path means at least two natural routes exist.
- high_distraction means the page likely contains realistic distractors, not just a verbose instruction.
- recovery means the wrong start is wrong-but-recoverable and recovery itself is a meaningful challenge.
- Prefer "revise" instead of "drop" when the core idea is good but the setup is weak.
- Use "unsure" when the task config alone is insufficient.

Return JSON only.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run text-only benchmark audit with a local instruct model.")
    parser.add_argument("--model-path", required=True, help="Local HF path or model id for a text instruct model.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Benchmark JSON input.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Prediction JSONL output.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--merged-csv",
        default="",
        help="Optional output CSV that merges original benchmark rows with llm_* columns.",
    )
    return parser.parse_args()


def load_benchmark(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"Expected list benchmark JSON, got {type(data)}")
    return data


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


def build_user_prompt(task: dict[str, Any]) -> str:
    eval_cfg = task.get("eval", {}) or {}
    lines = [
        f"benchmark_id: {task.get('benchmark_id', '')}",
        f"category: {task.get('category', '')}",
        f"site: {task.get('site', '')}",
        f"task_id: {task.get('task_id', task.get('source_task_id', ''))}",
        f"overall_difficulty: {task.get('overall_difficulty', '')}",
        "",
        "intent:",
        str(task.get("intent", "")),
        "",
        f"start_url: {task.get('start_url', '')}",
        f"wrong_start_url: {task.get('wrong_start_url', '')}",
        f"original_start_url: {task.get('original_start_url', '')}",
        f"require_login: {task.get('require_login', '')}",
        f"require_reset: {task.get('require_reset', '')}",
        "",
        "eval:",
        json.dumps(eval_cfg, ensure_ascii=False, indent=2),
    ]
    if task.get("construction_note"):
        lines.extend(["", "construction_note:", str(task["construction_note"])])
    return "\n".join(lines)


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


class TextJudge:
    def __init__(self, args: argparse.Namespace):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_map = {
            "auto": "auto",
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=dtype_map[args.torch_dtype],
            device_map=args.device_map,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
        )

    def generate(self, task: dict[str, Any]) -> tuple[dict[str, Any], str]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(task)},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = SYSTEM_PROMPT + "\n\n" + build_user_prompt(task)

        model_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        generated = self.model.generate(
            **model_inputs,
            max_new_tokens=self.args.max_new_tokens,
            do_sample=self.args.temperature > 0,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
        )
        output_ids = generated[0][model_inputs.input_ids.shape[1] :]
        raw_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        parsed = extract_json(raw_text)
        return parsed, raw_text


def normalize_record(task: dict[str, Any], parsed: dict[str, Any], raw_text: str, model_name: str) -> dict[str, Any]:
    return {
        "benchmark_id": task.get("benchmark_id"),
        "category": task.get("category"),
        "site": task.get("site"),
        "task_id": task.get("task_id", task.get("source_task_id")),
        "model_name": model_name,
        "parse_ok": True,
        "keep_or_drop": norm_choice(parsed.get("keep_or_drop"), {"keep", "drop", "revise"}, "revise"),
        "category_correct": norm_choice(parsed.get("category_correct"), {"yes", "no", "unsure"}, "unsure"),
        "multi_path_confirmed": norm_choice(parsed.get("multi_path_confirmed"), {"yes", "no", "unsure", "n/a"}, "n/a"),
        "distraction_visible": norm_choice(parsed.get("distraction_visible"), {"yes", "no", "unsure", "n/a"}, "n/a"),
        "recovery_setup_valid": norm_choice(parsed.get("recovery_setup_valid"), {"yes", "no", "unsure", "n/a"}, "n/a"),
        "recovery_severity": norm_choice(parsed.get("recovery_severity"), {"low", "medium", "high", "n/a"}, "n/a"),
        "task_solvable": norm_choice(parsed.get("task_solvable"), {"yes", "no", "unsure"}, "unsure"),
        "evaluator_reliable": norm_choice(parsed.get("evaluator_reliable"), {"yes", "no", "unsure"}, "unsure"),
        "duplicate_or_near_duplicate": norm_choice(parsed.get("duplicate_or_near_duplicate"), {"yes", "no", "unsure"}, "unsure"),
        "visual_dependence": norm_choice(parsed.get("visual_dependence"), {"high", "medium", "low"}, "medium"),
        "site_balance_priority": norm_choice(parsed.get("site_balance_priority"), {"high", "medium", "low"}, "medium"),
        "confidence": norm_confidence(parsed.get("confidence")),
        "rationale_short": str(parsed.get("rationale_short", "")),
        "cot_text": str(parsed.get("cot_text", "")),
        "raw_response": raw_text,
        "timestamp": datetime.now().isoformat(),
    }


def write_jsonl(path: Path, record: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_merged_csv(path: Path, benchmark: list[dict[str, Any]], prediction_path: Path) -> None:
    preds: dict[str, dict[str, Any]] = {}
    with open(prediction_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("benchmark_id"):
                preds[str(rec["benchmark_id"])] = rec

    rows = []
    for task in benchmark:
        row = dict(task)
        pred = preds.get(str(task.get("benchmark_id")), {})
        for key, value in pred.items():
            if key in {"raw_response", "cot_text"}:
                continue
            row[f"llm_{key}"] = value
        rows.append(row)

    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    benchmark = load_benchmark(Path(args.input))
    if args.start_index:
        benchmark = benchmark[args.start_index :]
    if args.limit is not None:
        benchmark = benchmark[: args.limit]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids = load_existing_predictions(output_path) if args.resume else set()
    judge = TextJudge(args)
    model_name = Path(args.model_path).name.rstrip("/") or args.model_path

    total = len(benchmark)
    for idx, task in enumerate(benchmark, start=1):
        benchmark_id = str(task.get("benchmark_id"))
        if benchmark_id in done_ids:
            continue
        try:
            parsed, raw_text = judge.generate(task)
            record = normalize_record(task, parsed, raw_text, model_name)
        except Exception as exc:
            record = {
                "benchmark_id": benchmark_id,
                "category": task.get("category"),
                "site": task.get("site"),
                "task_id": task.get("task_id", task.get("source_task_id")),
                "model_name": model_name,
                "parse_ok": False,
                "error": repr(exc),
                "timestamp": datetime.now().isoformat(),
            }
        write_jsonl(output_path, record)
        print(f"[{idx}/{total}] {benchmark_id} -> {record.get('keep_or_drop')} conf={record.get('confidence')}")

    if args.merged_csv:
        write_merged_csv(Path(args.merged_csv), load_benchmark(Path(args.input)), output_path)
        print(f"Merged CSV written to {args.merged_csv}")


if __name__ == "__main__":
    main()
