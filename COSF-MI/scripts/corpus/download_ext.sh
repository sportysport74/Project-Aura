#!/usr/bin/env bash
# Corpus extension for the 10B pretrain (experiments/plans/pretrain_10b.md): shards the 2.2B run never read, same three
# sources and licences as v1 (ODC-By collections; Cosmopedia text by Mixtral-8x7B-Instruct, Apache-2.0). Shard counts come
# from v1's measured yield (FineWeb-Edu 0.375B tokens/GB, Cosmopedia 0.244, FineMath 0.609) plus ~1 shard of slack each:
#   FineWeb-Edu sample/10BT 002-008 (7 × 2.0 GB)   Cosmopedia v2 00003-00012 (10 × 1.2 GB)   FineMath-4+ 00002-00007 (6 × 0.3 GB)
# ≈ 28 GB into data/corpus_v2/raw/<source>/ (same layout as v1). One file per call (two --include flags misparse).
# Usage (from COSF-MI/): bash scripts/corpus/download_ext.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
RAW=data/corpus_v2/raw
get() { [ -f "$RAW/$2/$3" ] || hf download "$1" "$3" --repo-type dataset --local-dir "$RAW/$2" > /dev/null; echo "ok $2/$3"; }
for i in 002 003 004 005 006 007 008; do get HuggingFaceFW/fineweb-edu fineweb_edu "sample/10BT/${i}_00000.parquet"; done
for i in $(seq 3 12); do get HuggingFaceTB/smollm-corpus cosmopedia_v2 "cosmopedia-v2/train-$(printf %05d "$i")-of-00104.parquet"; done
for i in $(seq 2 7); do get HuggingFaceTB/finemath finemath4 "finemath-4plus/train-$(printf %05d "$i")-of-00064.parquet"; done
du -sh "$RAW"/*
