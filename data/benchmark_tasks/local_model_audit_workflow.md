# Local Model Audit Workflow

This repository supports a two-stage benchmark curation workflow:

1. Text-first audit with a local instruct model
2. Visual re-audit with a local VLM on selected tasks

## Stage 1: Text Audit

Recommended model:

- `Qwen3-8B`

Example:

```bash
cd /root/autodl-tmp/Web-pqb

python scripts/benchmark_audit_with_llm.py \
  --model-path /root/autodl-tmp/web-models/models/Qwen3-8B \
  --input data/benchmark_tasks/webpqb_benchmark.json \
  --output data/benchmark_tasks/webpqb_benchmark_text_audit.jsonl \
  --merged-csv data/benchmark_tasks/webpqb_benchmark_text_audit.csv \
  --resume \
  --local-files-only
```

## Stage 2: Visual Re-audit

Recommended model:

- `Qwen2.5-VL-7B-Instruct`

Example:

```bash
cd /root/autodl-tmp/Web-pqb

python scripts/benchmark_reaudit_with_vlm.py \
  --model-path /root/autodl-tmp/web-models/models/Qwen2.5-VL-7B-Instruct \
  --benchmark data/benchmark_tasks/webpqb_benchmark.json \
  --text-audit data/benchmark_tasks/webpqb_benchmark_text_audit.jsonl \
  --render-tar data/_raw/gpt4v_som_910 \
  --output data/benchmark_tasks/webpqb_benchmark_visual_audit.jsonl \
  --select-mode flagged \
  --resume \
  --local-files-only
```

## Notes

- The visual re-audit script uses VisualWebArena render HTML files to recover a related screenshot.
- Recovery tasks use `source_task_id` to locate a related screenshot; the screenshot may come from the source task trajectory rather than the injected `wrong_start_url`.
- The final benchmark should still be human-curated after model pre-audit.
