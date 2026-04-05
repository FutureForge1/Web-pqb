#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve Qwen2.5-VL prelabel inference over HTTP.")
    parser.add_argument("--model-path", required=True, help="Local HF snapshot path on the GPU machine.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    think_start = text.find("<think>")
    think_end = text.rfind("</think>")
    if think_start != -1 and think_end != -1 and think_end > think_start:
        text = (text[:think_start] + text[think_end + len("</think>") :]).strip()

    first = text.find("{")
    last = text.rfind("}")
    if first == -1 or last == -1 or last <= first:
        raise ValueError("No JSON object found in model output.")
    return json.loads(text[first : last + 1])


class QwenServerJudge:
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
        self.model_name = Path(args.model_path).name.rstrip("/") or args.model_path

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        before_b64 = payload["before_image_b64"]
        after_b64 = payload["after_image_b64"]
        system_prompt = payload["system_prompt"]
        user_prompt = payload["user_prompt"]
        generation = payload.get("generation", {})

        before = Image.open(io.BytesIO(base64.b64decode(before_b64))).convert("RGB")
        after = Image.open(io.BytesIO(base64.b64decode(after_b64))).convert("RGB")

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": before},
                    {"type": "image", "image": after},
                    {"type": "text", "text": user_prompt},
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

        max_new_tokens = int(generation.get("max_new_tokens", 320))
        temperature = float(generation.get("temperature", 0.0))
        top_p = float(generation.get("top_p", 0.9))

        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": temperature,
            "top_p": top_p,
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
        return {
            "model_name": self.model_name,
            "parsed": parsed,
            "raw_text": raw_text,
        }


def make_handler(judge: QwenServerJudge):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/health":
                self.send_response(404)
                self.end_headers()
                return
            body = json.dumps({"status": "ok", "model_name": judge.model_name}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            if self.path != "/predict":
                self.send_response(404)
                self.end_headers()
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(length)
                payload = json.loads(raw_body.decode("utf-8"))
                result = judge.predict(payload)
                body = json.dumps(result, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
            except Exception as exc:
                body = json.dumps({"error": repr(exc)}, ensure_ascii=False).encode("utf-8")
                self.send_response(500)

            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def main() -> None:
    args = parse_args()
    judge = QwenServerJudge(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(judge))
    print(f"Serving {judge.model_name} on http://{args.host}:{args.port}")
    print(f"Health check: http://{args.host}:{args.port}/health")
    print("Predict endpoint: POST /predict")
    server.serve_forever()


if __name__ == "__main__":
    main()
