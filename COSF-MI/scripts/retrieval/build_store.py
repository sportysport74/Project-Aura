"""
Build a retrieval store's passages (experiments/plans/retrieval_prereg.md). CPU only.

  corpus  Cosmopedia v2 + FineWeb-Edu (int_score >= 3) from data/corpus_v1/raw (ODC-By), capped at --cap passages (uniform subsample)
  wiki    Simple English Wikipedia (wikimedia/wikipedia 20231101.simple, CC BY-SA 3.0 + GFDL; INFERENCE ONLY, never training data),
          title kept

Documents are cut into passages of <= --max-words words on sentence boundaries (shorter than --min-words are dropped). Leakage filter: any
passage containing a word 8-gram of a factuality reference answer is dropped. Writes <out>/passages.jsonl (id, source, title, text) and
<out>/build.json (counts, sources, licences).

Usage (from COSF-MI/): .venv/bin/python scripts/retrieval/build_store.py corpus --out data/retrieval/corpus [--max-docs N for smoke]
"""
import argparse
import glob
import json
import random
import re
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
SENT = re.compile(r"(?<=[.!?])\s+")
WORD = re.compile(r"[a-z0-9]+")


def chunks(text, max_words, min_words):
    for para in re.split(r"\n\s*\n|\n", text):
        cur = []
        for sent in SENT.split(para.strip()):
            w = sent.split()
            if not w:
                continue
            if cur and len(cur) + len(w) > max_words:
                if len(cur) >= min_words:
                    yield " ".join(cur)
                cur = []
            cur += w[:max_words]
        if len(cur) >= min_words:
            yield " ".join(cur)


def ngrams(text, n=8):
    w = WORD.findall(text.lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def docs(source, max_docs):
    if source == "corpus":
        specs = [("cosmopedia_v2", "data/corpus_v1/raw/cosmopedia_v2/**/*.parquet", None),
                 ("fineweb_edu", "data/corpus_v1/raw/fineweb_edu/**/*.parquet", "int_score")]
        for name, g, score in specs:
            n = 0
            for f in sorted(glob.glob(str(ROOT / g), recursive=True)):
                pf = pq.ParquetFile(f)
                cols = ["text"] + ([score] if score else [])
                for rg in range(pf.num_row_groups):
                    t = pf.read_row_group(rg, columns=cols).to_pydict()
                    for i, text in enumerate(t["text"]):
                        if score and (t[score][i] or 0) < 3:
                            continue
                        yield name, "", text
                        n += 1
                        if max_docs and n >= max_docs:
                            break
                    if max_docs and n >= max_docs:
                        break
                if max_docs and n >= max_docs:
                    break
    else:
        n = 0
        for f in sorted(glob.glob(str(ROOT / "data/retrieval/raw/wiki_simple/**/*.parquet"), recursive=True)):
            pf = pq.ParquetFile(f)
            for rg in range(pf.num_row_groups):
                t = pf.read_row_group(rg, columns=["title", "text"]).to_pydict()
                for title, text in zip(t["title"], t["text"]):
                    yield "wikipedia", title, text
                    n += 1
                    if max_docs and n >= max_docs:
                        return


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", choices=["corpus", "wiki"])
    p.add_argument("--out", required=True)
    p.add_argument("--max-words", type=int, default=90)
    p.add_argument("--min-words", type=int, default=20)
    p.add_argument("--cap", type=int, default=6_000_000)
    p.add_argument("--max-docs", type=int, default=0, help="per source, for smoke tests")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    leak = set()
    for line in open(ROOT / "experiments/factuality/questions.jsonl"):
        leak |= ngrams(json.loads(line)["reference"])
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    # stream every passage to a temp file (RAM holds no passages), then keep a uniform random subset of indices if over the cap
    dropped, ndocs, total = 0, {}, 0
    tmp = out / "all_passages.tmp.jsonl"
    with open(tmp, "w") as f:
        for src, title, text in docs(args.source, args.max_docs):
            ndocs[src] = ndocs.get(src, 0) + 1
            for c in chunks(text, args.max_words, args.min_words):
                if ngrams(c) & leak:
                    dropped += 1
                    continue
                f.write(json.dumps([src, title, c], ensure_ascii=False) + "\n")
                total += 1
    keep = None
    if args.source == "corpus" and total > args.cap:
        keep = set(random.Random(args.seed).sample(range(total), args.cap))
    n = 0
    with open(tmp) as fi, open(out / "passages.jsonl", "w") as fo:
        for i, line in enumerate(fi):
            if keep is not None and i not in keep:
                continue
            src, title, c = json.loads(line)
            fo.write(json.dumps({"id": n, "source": src, "title": title, "text": c}, ensure_ascii=False) + "\n")
            n += 1
    tmp.unlink()
    passages = range(n)
    lic = ({"cosmopedia_v2": "ODC-By 1.0; text by Mixtral-8x7B-Instruct (Apache-2.0)", "fineweb_edu": "ODC-By 1.0 (CommonCrawl web text)"}
           if args.source == "corpus" else {"wikipedia": "CC BY-SA 3.0 + GFDL (Simple English Wikipedia 20231101): INFERENCE ONLY"})
    info = {"source": args.source, "docs": ndocs, "passages_before_cap": total, "passages": len(passages), "leak_dropped": dropped,
            "max_words": args.max_words, "min_words": args.min_words, "cap": args.cap, "licences": lic}
    (out / "build.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
