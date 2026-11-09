#!/usr/bin/env bash
# Corpus ext2 for the Aura v3 base (experiments/plans/aura_v3_base.md, "Data"): shards no earlier corpus read (v1: 000-001 /
# 00000-00002 / 00000-00001; ext: FineWeb-Edu 002-008, Cosmopedia 00003-00012, FineMath 00002-00007). Same sources and licences:
#   FineWeb-Edu sample/10BT 009-013 (~2.8B tokens)   Cosmopedia v2 00013-00019 (~1.7B)   FineMath-4+ 00008-00011 (~0.5B)
# ≈ 19 GB into data/corpus_v3/raw/<source>/ (same layout as v1/v2). Re-runnable: finished files are skipped.
# Usage (from COSF-MI/): bash scripts/corpus/download_ext2.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
RAW=data/corpus_v3/raw
get() { [ -f "$RAW/$2/$3" ] || hf download "$1" "$3" --repo-type dataset --local-dir "$RAW/$2" > /dev/null; echo "$(date +%T) ok $2/$3"; }
for i in 009 010 011 012 013; do get HuggingFaceFW/fineweb-edu fineweb_edu "sample/10BT/${i}_00000.parquet"; done
for i in $(seq 13 19); do get HuggingFaceTB/smollm-corpus cosmopedia_v2 "cosmopedia-v2/train-$(printf %05d "$i")-of-00104.parquet"; done
for i in $(seq 8 11); do get HuggingFaceTB/finemath finemath4 "finemath-4plus/train-$(printf %05d "$i")-of-00064.parquet"; done
( cd "$RAW" && find . -name '*.parquet' -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS )
du -sh "$RAW"/*
