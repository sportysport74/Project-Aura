"""
Turn the downloaded prompt datasets into plain JSONL for the distillation pilot.

Inputs (downloaded once with `hf download`, each into its own folder under data/distill_sources/):
  OASST2 (OpenAssistant/oasst2, Apache-2.0): English root prompts by humans, 15-300 characters.
  GSM8K  (openai/gsm8k, MIT): train questions with their gold answers (the number after '####').

Outputs in data/distill_pilot/:
  src_oasst2.jsonl  {"source", "license", "prompt"}
  src_gsm8k.jsonl   {"source", "license", "prompt", "gold"}

Needs pandas + pyarrow (run it with ~/venvs/ml/bin/python -I). The vLLM venv only reads the JSONL.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sources", default=str(ROOT / "data/distill_sources"))
    p.add_argument("--out", default=str(ROOT / "data/distill_pilot"))
    args = p.parse_args()
    src, out = Path(args.sources), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    o = pd.read_parquet(next((src / "oasst2").rglob("train-*.parquet")),
                        columns=["text", "role", "lang", "parent_id", "deleted"])
    roots = o[o.parent_id.isna() & (o.role == "prompter") & (o.lang == "en") & ~o.deleted.fillna(False)]
    roots = roots[roots.text.str.len().between(15, 300)].drop_duplicates("text")
    with open(out / "src_oasst2.jsonl", "w") as f:
        for t in roots.text:
            f.write(json.dumps({"source": "oasst2", "license": "Apache-2.0", "prompt": t.strip()}) + "\n")

    g = pd.read_parquet(next((src / "gsm8k").rglob("train-*.parquet")))
    with open(out / "src_gsm8k.jsonl", "w") as f:
        for q, a in zip(g.question, g.answer):
            gold = a.split("####")[-1].strip().replace(",", "")
            f.write(json.dumps({"source": "gsm8k", "license": "MIT", "prompt": q.strip(), "gold": gold}) + "\n")
    print(f"OASST2: {len(roots):,} English root prompts -> {out / 'src_oasst2.jsonl'}")
    print(f"GSM8K: {len(g):,} questions -> {out / 'src_gsm8k.jsonl'}")


if __name__ == "__main__":
    main()
