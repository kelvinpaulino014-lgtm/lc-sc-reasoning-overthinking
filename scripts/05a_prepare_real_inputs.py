"""Freeze question-disjoint development and evaluation lists from local dataset snapshots."""
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/phase4"
HUB = Path.home() / ".cache/huggingface/hub"
GSM = HUB / "datasets--openai--gsm8k/snapshots/740312add88f781978c0658806c59bc2815b9866/main"
MATH = HUB / "datasets--HuggingFaceH4--math-500/snapshots/6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be/test.jsonl"
SEED = 24112026


def key(question):
    return " ".join(question.casefold().split())


def gsm_rows(split):
    frame = pd.read_parquet(GSM / f"{split}-00000-of-00001.parquet")
    for i, row in frame.iterrows():
        reasoning, answer = row.answer.rsplit("####", 1)
        level = min(5, max(1, len(reasoning.strip().splitlines()) - 1))
        yield {"question_id": f"gsm8k_{split}_{i:04d}", "dataset": "gsm8k",
               "question": row.question.strip(), "answer_gold": answer.strip().replace(",", ""),
               "stratum": level}


def math_rows():
    for line in MATH.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        yield {"question_id": "math500_" + row["unique_id"], "dataset": "math500",
               "question": row["problem"].strip(), "answer_gold": row["answer"],
               "stratum": int(re.search(r"\d+", str(row["level"])).group())}


def draw(pool, n, seed):
    selected, rest = train_test_split(pool, train_size=n, random_state=seed,
                                      stratify=[r["stratum"] for r in pool])
    return selected, rest


def write_once(name, data):
    path = OUT / name
    encoded = data.encode("utf-8")
    if path.exists() and path.read_bytes() != encoded:
        raise ValueError(f"Different frozen data already exist at {path}")
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    seen = {key(json.loads(line)["question"]) for infile in (ROOT / "data").glob("*_splits.jsonl")
            for line in infile.read_text(encoding="utf-8").splitlines()}

    def fresh(records):
        rows = []
        for row in records:
            text = key(row["question"])
            if text not in seen:
                seen.add(text)
                rows.append(row)
        return rows

    gsm_test = fresh(gsm_rows("test"))
    gsm_train = fresh(gsm_rows("train"))
    maths = fresh(math_rows())
    gsm_holdout, _ = draw(gsm_test, 96, SEED)
    gsm_fit, gsm_remaining = draw(gsm_train, 80, SEED + 1)
    gsm_cal, _ = draw(gsm_remaining, 32, SEED + 2)
    math_holdout, math_remaining = draw(maths, 96, SEED)
    math_fit, math_remaining = draw(math_remaining, 80, SEED + 1)
    math_cal, _ = draw(math_remaining, 32, SEED + 2)
    inputs, labels = [], []
    for split, a, b in [("train", gsm_fit, math_fit), ("calibration", gsm_cal, math_cal),
                        ("holdout", gsm_holdout, math_holdout)]:
        for pair in zip(a, b):
            for row in pair:
                common = {"question_id": row["question_id"], "dataset": row["dataset"], "split": split}
                inputs.append(dict(common, question=row["question"]))
                labels.append(dict(common, answer_gold=row["answer_gold"], stratum=row["stratum"]))
    hashes = {}
    for name, records in [("inputs.jsonl", inputs), ("labels.jsonl", labels)]:
        hashes[name] = write_once(name, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    write_once("split_manifest.json", json.dumps(dict(seed=SEED, sha256=hashes,
               gsm_snapshot=GSM.parent.name, math_snapshot=MATH.parent.name,
               per_dataset={"train": 80, "calibration": 32, "holdout": 96}), indent=2))
    print(len(inputs), "distinct questions:", len({key(r["question"]) for r in inputs}), hashes)


if __name__ == "__main__":
    main()
