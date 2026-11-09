#!/usr/bin/env bash
# Corpus ext2 for the Aura v3 base (experiments/plans/aura_v3_base.md, "Data"), CPU only: download (download_ext2.sh), arithmetic
# (new seed 3, every non-fact problem of arith_corpus_v1/v2 excluded), tokenize to data/corpus_v3/tok24k_ext2 (~5.0B tokens,
# shares 55 / 30 / 10 / 3.5%, no distill). Each finished piece is skipped on a rerun. Alerts by ntfy at the end or on failure.
# Usage (from COSF-MI/): nohup bash scripts/corpus/build_ext2.sh > data/corpus_v3/build.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
trap 'bash scripts/notify.sh "corpus ext2 build FAILED (data/corpus_v3/build.log)" 5' ERR
PY=.venv/bin/python
bash scripts/corpus/download_ext2.sh
if [ ! -f data/arith_corpus_v3/report.json ]; then
  $PY scripts/distill/build_arith_plain.py --n 4400000 --seed 3 --out data/arith_corpus_v3 --exclude-jsonl \
    data/arith_corpus_v1/train.jsonl data/arith_corpus_v1/val.jsonl data/arith_corpus_v2/train.jsonl data/arith_corpus_v2/val.jsonl
fi
if [ ! -f data/corpus_v3/tok24k_ext2/meta.json ]; then
  $PY scripts/corpus/build_corpus.py tokenize --tokenizer models/tokenizer_clean24k.model --tokens 5.08e9 --name tok24k_ext2 \
    --raw data/corpus_v3/raw --root data/corpus_v3 --shares aura_distill:0 --arith-jsonl data/arith_corpus_v3/train.jsonl
fi
$PY -c "import json;m=json.load(open('data/corpus_v3/tok24k_ext2/meta.json'));print('train', m['train_tokens'], {k: v['train_tokens'] for k, v in m['sources'].items()})"
bash scripts/notify.sh "corpus ext2 built: data/corpus_v3/tok24k_ext2" 3
echo "=== $(date -Iseconds) EXT2_DONE"
