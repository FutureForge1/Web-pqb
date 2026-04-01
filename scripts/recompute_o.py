#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute O_t / R_t from V_t and y_t fields.")
    parser.add_argument("--input", required=True, help="Input JSONL or parquet path.")
    parser.add_argument("--output", required=True, help="Output JSONL or parquet path.")
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--v-key", default="V_t")
    parser.add_argument("--y-key", default="y_t")
    parser.add_argument("--o-prev-key", default="O_prev")
    parser.add_argument("--o-key", default="O_t")
    parser.add_argument("--r-key", default="R_t")
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records
    if path.suffix == ".parquet":
        return pd.read_parquet(path).to_dict("records")
    raise SystemExit(f"Unsupported input format: {path.suffix}")


def save_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jsonl":
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return
    if path.suffix == ".parquet":
        pd.DataFrame(records).to_parquet(path, index=False)
        return
    raise SystemExit(f"Unsupported output format: {path.suffix}")


def coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    records = load_records(input_path)
    records.sort(
        key=lambda rec: (
            str(rec.get("source", "")),
            str(rec.get("trajectory_id", rec.get("task_id", ""))),
            int(rec.get("step_idx", 0)),
        )
    )

    current_o_by_traj: dict[tuple[str, str], float] = {}
    updated: list[dict[str, Any]] = []

    for rec in records:
        source = str(rec.get("source", ""))
        trajectory_id = str(rec.get("trajectory_id", rec.get("task_id", "")))
        traj_key = (source, trajectory_id)
        prev_o = current_o_by_traj.get(traj_key, 1.0)

        v_t = coerce_int(rec.get(args.v_key))
        y_t = coerce_int(rec.get(args.y_key))

        if y_t is None:
            o_t = None
            r_t = None
        else:
            o_t = 1.0 if y_t == 1 else args.gamma * prev_o
            r_t = (v_t * o_t) if v_t is not None else None
            current_o_by_traj[traj_key] = o_t

        rec[args.o_prev_key] = prev_o
        rec[args.o_key] = o_t
        rec[args.r_key] = r_t
        updated.append(rec)

    save_records(output_path, updated)
    print(f"Wrote {len(updated)} records -> {output_path}")


if __name__ == "__main__":
    main()
