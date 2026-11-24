"""Collect independently sampled Qwen continuations, one durable question at a time."""
import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "phase4"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
PROMPT = "You are a precise math assistant. Solve the problem step by step and put your final answer inside \\boxed{}."
CONFIG = dict(num_return_sequences=4, max_new_tokens=384, do_sample=True, temperature=0.7,
              top_p=0.95, top_k=20, repetition_penalty=1.1)
SCHEMA = "phase4-real-v1"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def question_seed(row):
    key = f"{SCHEMA}|{row['dataset']}|{row['question_id']}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")


class Deadline(StoppingCriteria):
    def __init__(self, until):
        self.until = until
        self.expired = False

    def __call__(self, input_ids, scores, **kwargs):
        self.expired = time.time() >= self.until
        return self.expired


@contextmanager
def writer_lock(path):
    # The OS releases the lock after a process crash.
    with open(path, "a+b") as lock:
        lock.seek(0, 2)
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def load_inputs(path):
    data = path.read_bytes()
    rows = [json.loads(line) for line in data.splitlines()]
    if not rows:
        raise ValueError("An empty question list cannot be collected.")
    allowed = {"question_id", "dataset", "split", "question"}
    if any(set(row) != allowed or not all(isinstance(v, str) and v for v in row.values()) for row in rows):
        raise ValueError("Inputs must contain exactly the four nonempty blind fields.")
    if len({row["question_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate question_id.")
    return rows, digest(data)


def recover_records(path, rows):
    if not path.exists():
        return 0
    n = 0
    with path.open("r+b") as f:
        while True:
            position = f.tell()
            line = f.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                f.truncate(position)
                f.flush()
                os.fsync(f.fileno())
                print("Recovered interrupted trailing record.", flush=True)
                break
            record = json.loads(line)
            if n >= len(rows) or any(record[k] != rows[n][k] for k in ("question_id", "dataset", "split")):
                raise ValueError(f"Saved record order or split differs at {n}.")
            if record["seed"] != question_seed(rows[n]) or len(record["traces"]) != CONFIG["num_return_sequences"]:
                raise ValueError(f"Saved seed or trace count differs at {n}.")
            n += 1
    return n


def save_manifest(path, manifest):
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old["protocol"] != manifest["protocol"]:
            raise RuntimeError("Existing collection has different inputs, settings or software.")
        return
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


@torch.inference_mode()
def collect_one(row, model, tok, until):
    seed = question_seed(row)
    torch.manual_seed(seed)
    messages = [{"role": "system", "content": PROMPT}, {"role": "user", "content": row["question"].strip()}]
    ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
    deadline = Deadline(until)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), **CONFIG,
                         eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
                         output_scores=True, return_dict_in_generate=True,
                         stopping_criteria=StoppingCriteriaList([deadline]))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    if deadline.expired:
        return None
    generated = out.sequences[:, ids.shape[1]:]
    if generated.shape[1] != len(out.scores):
        raise ValueError("Number of stored scores differs from generation steps.")
    # After top-k/top-p, only at most 20 logits can typically be finite. Ties at
    # the top-k cutoff can keep more, so compute exact entropy on the full vector.
    chunks_h, chunks_p = [], []
    for start in range(0, len(out.scores), 4):
        scores = torch.stack(out.scores[start:start + 4]).float()
        log_p = scores.log_softmax(dim=-1)
        probs = log_p.exp()
        picked = generated[:, start:start + len(scores)].T.unsqueeze(-1)
        chunks_h.append(torch.special.entr(probs).sum(dim=-1))
        chunks_p.append(log_p.gather(-1, picked).squeeze(-1))
    entropy = torch.cat(chunks_h).T.tolist()
    logprob = torch.cat(chunks_p).T.tolist()
    traces = []
    for i, sequence in enumerate(generated.tolist()):
        eos = tok.eos_token_id in sequence
        n = sequence.index(tok.eos_token_id) + 1 if eos else len(sequence)
        if not eos and n != CONFIG["max_new_tokens"]:
            raise ValueError("An unfinished sequence is shorter than the declared token budget.")
        if eos and any(t != tok.pad_token_id for t in sequence[n:]):
            raise ValueError("Unexpected token after first EOS.")
        h, lp = entropy[i][:n], logprob[i][:n]
        if not all(math.isfinite(v) for v in h + lp) or min(h) < -1e-5 or max(lp) > 1e-5:
            raise ValueError("Invalid token telemetry.")
        token_ids = sequence[:n]
        traces.append({"sample_idx": i, "token_ids": token_ids, "text": tok.decode(token_ids, skip_special_tokens=True),
                       "entropies": h, "logprobs": lp, "eos": eos})
    return {"schema": SCHEMA, **{k: row[k] for k in ("question_id", "dataset", "split")}, "seed": seed,
            "elapsed_sec": elapsed, "collection_sec": time.perf_counter() - t0, "prompt_tokens": ids.shape[1],
            "batch_steps": generated.shape[1], "recorded_utc": utc_now(), "traces": traces}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deadline", required=True, help="Absolute ISO-8601 datetime, with time zone")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    dt = datetime.fromisoformat(args.deadline)
    if dt.tzinfo is None:
        ap.error("An explicit timezone is required.")
    until = dt.timestamp()
    if time.time() >= until:
        raise SystemExit("Deadline already reached.")
    rows, input_sha = load_inputs(args.out / "inputs.jsonl")
    model_path = snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    device = torch.cuda.get_device_properties(0)
    protocol = {"schema": SCHEMA, "model": MODEL_ID, "revision": REVISION, "dtype": "float16",
                "device": device.name, "inputs_sha256": input_sha, "n_inputs": len(rows), "generation": CONFIG,
                "eos_token_id": tok.eos_token_id, "pad_token_id": tok.pad_token_id, "prompt": PROMPT,
                "chat_template_sha256": digest(tok.chat_template.encode()),
                "score_definition": "Entropy (nats) and sampled log probability after warpers and logits processors",
                "seed_definition": "uint32 big-endian of SHA256('phase4-real-v1|dataset|question_id')[:4]",
                "torch": str(torch.__version__), "cuda": torch.version.cuda, "transformers": transformers.__version__,
                "python": sys.version.split()[0]}
    manifest = {"protocol": protocol, "created_utc": utc_now(), "platform": platform.platform(),
                "script_sha256": digest(Path(__file__).read_bytes()), "deadline": dt.isoformat(),
                "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    path = args.out / "traces.jsonl"
    with writer_lock(args.out / "collection.lock"):
        if path.exists() and path.stat().st_size and not (args.out / "collection_manifest.json").is_file():
            raise RuntimeError("Existing data have no manifest.")
        save_manifest(args.out / "collection_manifest.json", manifest)
        done = recover_records(path, rows)
        if done == len(rows):
            print(f"Complete: {done} questions.", flush=True)
            return
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16,
                                                     device_map="cuda", local_files_only=True).eval()
        with path.open("ab", buffering=0) as output:
            for i, row in enumerate(rows[done:], start=done + 1):
                if time.time() >= until:
                    break
                result = collect_one(row, model, tok, until)
                if result is None:
                    break
                line = (json.dumps(result, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode()
                if output.write(line) != len(line):
                    raise OSError("Short trace record write.")
                os.fsync(output.fileno())
                lengths = [len(s["token_ids"]) for s in result["traces"]]
                print(f"{utc_now()} {i}/{len(rows)} {row['question_id']} tokens={lengths} "
                      f"generation={result['elapsed_sec']:.2f}s", flush=True)
                done = i
        print(f"Saved {done}/{len(rows)} questions.", flush=True)


if __name__ == "__main__":
    main()

