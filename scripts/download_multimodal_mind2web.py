#!/usr/bin/env python3

import os
import time
from pathlib import Path

import requests


ROOT = Path("data/multimodal_mind2web/data")
MAX_WORKERS = 1
TIMEOUT = 60
RETRIES = 6
CHUNK_SIZE = 1024 * 1024
BASE_URL = "https://huggingface.co/datasets/osunlp/Multimodal-Mind2Web/resolve/main/data/"
ROUND_SLEEP = 90

# File list mirrored from the official Hugging Face files page.
FILES = [
    "test_domain-00000-of-00011-26c55c12cbbcdc8e.parquet",
    "test_domain-00001-of-00011-93dadb8d3ca8a3e9.parquet",
    "test_domain-00002-of-00011-f4d93275c87bbd81.parquet",
    "test_domain-00003-of-00011-8e9c851b71133773.parquet",
    "test_domain-00004-of-00011-d94e067efdd549d4.parquet",
    "test_domain-00005-of-00011-f1164b95bfaed9de.parquet",
    "test_domain-00006-of-00011-a903b67c9fda87e5.parquet",
    "test_domain-00007-of-00011-5b3f5bd69a725501.parquet",
    "test_domain-00008-of-00011-555923175d587f8c.parquet",
    "test_domain-00009-of-00011-8c95b9e67a3679f9.parquet",
    "test_domain-00010-of-00011-4ec618cf686bc066.parquet",
    "test_task-00000-of-00005-431389419142b606.parquet",
    "test_task-00001-of-00005-bdd2cef984845c42.parquet",
    "test_task-00002-of-00005-e2696e51f1c78db8.parquet",
    "test_task-00003-of-00005-944419f61d7cf1fb.parquet",
    "test_task-00004-of-00005-a3a487e53da307b4.parquet",
    "test_website-00000-of-00004-e0bfff7049abbef8.parquet",
    "test_website-00001-of-00004-b0b6abaa088e90d8.parquet",
    "test_website-00002-of-00004-11198eb4fc38a82b.parquet",
    "test_website-00003-of-00004-3019191fc3984b5e.parquet",
    "train-00000-of-00027-4d11798d7219186d.parquet",
    "train-00001-of-00027-2011d8e72a165f62.parquet",
    "train-00002-of-00027-81107b64e8a3a046.parquet",
    "train-00003-of-00027-b2bcc4d20fbfb47d.parquet",
    "train-00004-of-00027-82ef57959581c455.parquet",
    "train-00005-of-00027-5880578c1eba7822.parquet",
    "train-00006-of-00027-94de4ed04ae0b588.parquet",
    "train-00007-of-00027-ae4267e946757225.parquet",
    "train-00008-of-00027-a361767e10599c01.parquet",
    "train-00009-of-00027-6389d25433f33aeb.parquet",
    "train-00010-of-00027-dc9d9f0049e1567c.parquet",
    "train-00011-of-00027-d307a333ead09969.parquet",
    "train-00012-of-00027-e10fbc31995f4638.parquet",
    "train-00013-of-00027-f12808857fc81595.parquet",
    "train-00014-of-00027-ed3e2dfdf0245d74.parquet",
    "train-00015-of-00027-3e1bf87116e5d714.parquet",
    "train-00016-of-00027-89dd493e8abfbe97.parquet",
    "train-00017-of-00027-2c7d6427a32411c4.parquet",
    "train-00018-of-00027-59871098159f2152.parquet",
    "train-00019-of-00027-55293ecc88d419ef.parquet",
    "train-00020-of-00027-a3f17abfa6315328.parquet",
    "train-00021-of-00027-52891bb870366cec.parquet",
    "train-00022-of-00027-12501ae3fc70271f.parquet",
    "train-00023-of-00027-9cb87a52ffbec4a9.parquet",
    "train-00024-of-00027-c75dad7737630204.parquet",
    "train-00025-of-00027-903fd897f9bfb606.parquet",
    "train-00026-of-00027-fdc31d0e2d56bf2f.parquet",
]


def download_one(name: str) -> tuple[str, str, str | int]:
    ROOT.mkdir(parents=True, exist_ok=True)
    out = ROOT / name
    tmp = ROOT / f"{name}.part"
    if out.exists() and not tmp.exists():
        return "skip", name, out.stat().st_size

    url = BASE_URL + name
    last_error: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            existing = tmp.stat().st_size if tmp.exists() else 0
            headers = {"User-Agent": "Mozilla/5.0", "Connection": "close"}
            mode = "wb"
            if existing > 0:
                headers["Range"] = f"bytes={existing}-"
                mode = "ab"

            with requests.get(url, headers=headers, stream=True, timeout=TIMEOUT) as resp:
                resp.raise_for_status()
                if existing and resp.status_code != 206:
                    existing = 0
                    mode = "wb"
                content_length = int(resp.headers.get("Content-Length", "0"))
                expected_size = existing + content_length if resp.status_code == 206 else content_length
                with open(tmp, mode) as fh:
                    for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            fh.write(chunk)

            size = tmp.stat().st_size
            if expected_size == 0 or size == expected_size:
                os.replace(tmp, out)
                return "ok", name, size
            if expected_size and size > expected_size:
                tmp.unlink(missing_ok=True)
            raise RuntimeError(f"size mismatch: {size} != {expected_size}")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(min(2 * attempt, 10))

    return "fail", name, str(last_error)


def main() -> int:
    start = time.time()
    completed = len([name for name in FILES if (ROOT / name).exists()])
    round_idx = 0
    print(f"total={len(FILES)} already_done={completed}", flush=True)

    while True:
        pending = [name for name in FILES if not (ROOT / name).exists()]
        if not pending:
            print("done_all", flush=True)
            return 0

        round_idx += 1
        print(
            f"round={round_idx} pending={len(pending)} completed={len(FILES) - len(pending)} elapsed={int(time.time() - start)}s",
            flush=True,
        )

        round_success = 0
        for idx, name in enumerate(pending, start=1):
            status, _, extra = download_one(name)
            elapsed = int(time.time() - start)
            print(f"[round {round_idx} {idx}/{len(pending)}] {status} {name} {extra} elapsed={elapsed}s", flush=True)
            if status == "ok":
                round_success += 1

        if round_success == 0:
            print(f"round={round_idx} no_progress sleep={ROUND_SLEEP}s", flush=True)
        else:
            print(f"round={round_idx} progress={round_success} sleep={ROUND_SLEEP}s", flush=True)
        time.sleep(ROUND_SLEEP)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
