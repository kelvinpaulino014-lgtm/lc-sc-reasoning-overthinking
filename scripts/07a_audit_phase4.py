"""Periodic, read-only scientific integrity audit of the local experiment."""
import argparse
import hashlib
import json
import math
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/phase4"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lines(path):
    with path.open("rb") as handle:
        for data in handle:
            if data.endswith(b"\n"):
                yield json.loads(data)


def audit():
    errors = []
    counts = Counter()
    totals = Counter()
    manifest = json.loads((OUT / "split_manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        if sha(OUT / name) != expected:
            errors.append(f"Changed frozen inputs: {name}")
    inputs = list(lines(OUT / "inputs.jsonl"))
    ids = {r["question_id"] for r in inputs}
    if len(ids) != len(inputs) or any(set(r) != {"dataset", "question_id", "split", "question"} for r in inputs):
        errors.append("Duplicate IDs or unexpected fields in blind input")
    history = {" ".join(r["question"].casefold().split()) for f in (ROOT / "data").glob("*_splits.jsonl")
               for r in lines(f)}
    present = [" ".join(r["question"].casefold().split()) for r in inputs]
    if len(set(present)) != len(present) or set(present) & history:
        errors.append("Overlap with previous questions or within current splits")
    path = OUT / "traces.jsonl"
    clock = []
    if path.exists():
        for i, row in enumerate(lines(path)):
            if i >= len(inputs) or any(row[k] != inputs[i][k] for k in ("dataset", "split", "question_id")):
                errors.append(f"Out-of-order or unknown row {i}")
                continue
            if "answer_gold" in row or len(row["traces"]) != 4:
                errors.append(f"Labels or missing samples in blind row {i}")
            counts[f'{row["dataset"]}/{row["split"]}'] += 1
            clock.append(row["elapsed_sec"])
            for tr in row["traces"]:
                n = len(tr["token_ids"])
                if (not 1 <= n <= 384 or len(tr["entropies"]) != n or len(tr["logprobs"]) != n
                        or not all(math.isfinite(v) and v >= -1e-5 for v in tr["entropies"])
                        or not all(math.isfinite(v) and v <= 1e-5 for v in tr["logprobs"])
                        or tr["eos"] != (tr["token_ids"][-1] == 151645)
                        or (not tr["eos"] and n != 384)):
                    errors.append(f"Invalid token telemetry row {i}")
                totals["traces"] += 1
                totals["tokens"] += n
                totals["capped"] += int(not tr["eos"])
    live = OUT / "online" / "observations.jsonl"
    live_counts = Counter(row["method"] for row in lines(live)) if live.exists() else {}
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,utilization.gpu",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
    return {"utc": datetime.now(timezone.utc).isoformat(), "questions": sum(counts.values()),
            "counts": dict(counts), **totals, "online_method_counts": dict(live_counts),
            "mean_last_10_generation_sec": round(sum(clock[-10:]) / len(clock[-10:]), 2) if clock else None,
            "gpu_temperature_C_memory_MiB_util_pct": gpu.stdout.strip(), "errors": errors,
            "scope": "provenance, forbidden input fields, split overlap, IDs, causal telemetry lengths, EOS, finite values; no holdout outcomes inspected"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", help="Timezone-aware ISO start, for a 20-minute cadence")
    p.add_argument("--until", help="Timezone-aware ISO deadline")
    p.add_argument("--once", action="store_true")
    args = p.parse_args()
    due = time.time() if args.once else datetime.fromisoformat(args.start).timestamp()
    until = float("inf") if args.once else datetime.fromisoformat(args.until).timestamp()
    if not args.once:
        due += max(1, math.ceil((time.time() - due) / 1200)) * 1200
    while due <= until and not (OUT / "CLOSED.json").exists():
        if time.time() < due:
            time.sleep(min(30, due - time.time()))
            continue
        try:
            record = audit()
        except Exception as exc:
            record = {"utc": datetime.now(timezone.utc).isoformat(), "errors": [repr(exc)]}
        with (OUT / "audits.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if args.once:
            break
        due += 1200


if __name__ == "__main__":
    main()
