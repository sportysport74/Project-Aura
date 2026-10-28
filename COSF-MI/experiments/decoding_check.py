"""
How much of the chat looping is the decoding and how much the model? Re-runs the A/B's 40-prompt readable set and the
~400-prompt repetition stats (same prompts as ab_eval.py) under three sampling setups, 2 samples per prompt:
  topk40        temperature 0.7, top-k 40 (the A/B report's "sampled")
  topp          temperature 0.7, top-p 0.9
  topp_rep1.15  temperature 0.7, top-p 0.9, repetition penalty 1.15
Greedy numbers come from the A/B JSON. Output: <out>/<name>.json, summary.md, readable.md (topp_rep1.15, both samples).

Usage (from COSF-MI/):
  .venv/bin/python experiments/decoding_check.py --out experiments/results/ab_distill/decoding NAME=CKPT [...]
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
import ab_eval as ab  # noqa: E402

CONFIGS = {"topk40": dict(top_k=40, top_p=None, rep_penalty=1.0),
           "topp": dict(top_k=0, top_p=0.9, rep_penalty=1.0),
           "topp_rep1.15": dict(top_k=0, top_p=0.9, rep_penalty=1.15)}
SAMPLES = (1, 2)
METRICS = [("repeated_4gram", "replies with a repeated 4-gram", "{:.0%}"), ("loop_4gram_3x", "replies with a 4-gram 3+ times", "{:.0%}"),
           ("distinct2_mean", "distinct-2 per reply", "{:.3f}"), ("mean_words", "mean words", "{:.0f}"),
           ("turn_ended", "turn ended", "{:.0%}"), ("ends_with_question_convo", 'ends in "?" (chat/explain)', "{:.0%}")]


def run(name, path, rprompts, out_dir, bs=64):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = ab.COSFOscillatorTransformer(ckpt["config"]).to("cuda")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    prompts = [p for _, p in rprompts]
    res = {"name": name, "checkpoint": str(path), "stats": {}, "readable": {}}
    for cfg, kw in CONFIGS.items():
        rows, readable = [], []
        for s in SAMPLES:
            out = ab.generate(model, ab.tok_global, prompts, greedy=False, temperature=0.7, seed=s, bs=bs, **kw)
            rows += [(k, r, e) for (k, _), (r, e) in zip(rprompts, out)]
            readable.append([r for r, _ in out[:40]])
        res["stats"][cfg] = ab.text_stats(rows)
        res["readable"][cfg] = readable
        print(f"[{name}] {cfg}: repeated 4-gram {res['stats'][cfg]['repeated_4gram']:.0%}, "
              f"mean words {res['stats'][cfg]['mean_words']:.0f}", flush=True)
    (out_dir / f"{name}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    del model
    torch.cuda.empty_cache()
    return res


def report(out_dir, names, ab_dir):
    res = {n: json.loads((out_dir / f"{n}.json").read_text()) for n in names}
    greedy = {n: json.loads((ab_dir / f"{n}.json").read_text())["rates"]["greedy"] for n in names}
    L = ["# Decoding check: is the looping the decoding or the model?\n",
         "Same ~400 prompts as the A/B report; sampled rows pool 2 samples per prompt. Greedy is from the A/B run.\n"]
    for key, label, f in METRICS:
        L.append(f"\n## {label}\n\n| checkpoint | greedy | " + " | ".join(CONFIGS) + " |\n|---|---|" + "---|" * len(CONFIGS))
        for n in names:
            L.append(f"| {n} | {f.format(greedy[n][key])} | "
                     + " | ".join(f.format(res[n]["stats"][c][key]) for c in CONFIGS) + " |")
    (out_dir / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    readable = ab.READABLE
    prompts = [p for c in ab.RCATS for p in readable[c]]
    M = ["# Readable set, top-p 0.9, temperature 0.7, repetition penalty 1.15 (2 samples)\n"]
    for i, p in enumerate(prompts):
        M.append(f"\n### {i + 1}. {p}\n\n| checkpoint | sample 1 | sample 2 |\n|---|---|---|")
        for n in names:
            s1, s2 = (res[n]["readable"]["topp_rep1.15"][k][i].replace("\n", "<br>").replace("|", "\\|") for k in (0, 1))
            M.append(f"| {n} | {s1} | {s2} |")
    (out_dir / "readable.md").write_text("\n".join(M) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpts", nargs="+", help="NAME=path/to/best_model.pt")
    p.add_argument("--out", default="experiments/results/ab_distill/decoding")
    p.add_argument("--ab-dir", default="experiments/results/ab_distill")
    p.add_argument("--pilot", default="data/distill_pilot/ft_v1")
    p.add_argument("--bs", type=int, default=16, help="generation batch; 64 thrashed GPU memory (full-context logits) "
                   "and ended in CUDA_ERROR_UNKNOWN on 2026-10-28")
    args = p.parse_args()
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    ab.tok_global = ab.BPETokenizer(str(ROOT / "models/tokenizer_mix16k.model"))
    _, chat_val_lines, pilot_by_cat = ab.frozen_sets(ab.tok_global, ROOT / args.pilot)
    rprompts = ab.rate_prompts(chat_val_lines, pilot_by_cat)
    names = []
    for spec in args.ckpts:
        name, path = spec.split("=", 1)
        run(name, ROOT / path, rprompts, out_dir, args.bs)
        names.append(name)
    report(out_dir, names, ROOT / args.ab_dir)
    print(f"wrote {out_dir / 'summary.md'} and {out_dir / 'readable.md'}")


if __name__ == "__main__":
    main()
