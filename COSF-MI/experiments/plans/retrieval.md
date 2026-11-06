# Plan: retrieval for AURA (PLANNED 2026-11-05, CPU-only planning; nothing runs before the user approves)

**Why:** the 104M base knows few facts (chat factuality 0.30 after fine-tuning; 3-10% fully correct), and more pretraining tokens barely
move the probe. Retrieval puts the facts in a store and teaches AURA to read them before answering, instead of storing them in weights.

## 1. Fact store

| option | source | licence (checked on the Hub 2026-11-05) | size | recommendation |
|---|---|---|---|---|
| **S1 (default)** | Cosmopedia v2 + FineWeb-Edu, the shards already on disk (`data/corpus_v1/raw`: 1.13M + 1.46M docs) | ODC-By (collections; attribute); Cosmopedia text by Mixtral (Apache-2.0) | ~2.6M docs → **~6M passages** of ~100 tokens (sentence-aligned; keep FineWeb-Edu docs with edu score ≥ 3) | **Use.** No new licence and no download; textbook/explainer style that matches the eval questions |
| S2 (optional) | Simple English Wikipedia (`wikimedia/wikipedia`, `20231101.simple`, ~250k articles) | **CC BY-SA 3.0 + GFDL (share-alike)** | ~1M passages, ~0.3 GB parquet | **Flag (Nodal Systems):** fine as an *inference-time* store if shown passages are attributed and the store stays CC BY-SA; **keep it out of training data**, because SA's reach into trained weights is unsettled. Your call |
| S3 (later) | the same plus FineMath (ODC-By) for maths word facts | ODC-By | +~1M passages | only if the arithmetic/word-problem eval needs it |

Leakage check: the 100 factuality questions and reference answers (written for this eval) aren't in any source; the store builder still drops
any passage that contains a reference answer verbatim (≥ 20-token overlap).

## 2. Embedding index

- **Embedder: `BAAI/bge-small-en-v1.5`** (33M params, **MIT**), 384-dim, query prefix per its card. Alternatives checked: `intfloat/e5-small-v2`
  (MIT), `Snowflake/snowflake-arctic-embed-s` (Apache-2.0), `sentence-transformers/all-MiniLM-L6-v2` (Apache-2.0). Loaded with `transformers`
  (already in `~/venvs/ml`): no new dependency.
- **Index: brute-force inner product in torch** (fp16 matrix, 6M × 384 = 4.6 GB): on the GPU a batch of 200k queries takes seconds; on the CPU
  ~0.5-1 s per query (fine for eval and chat). No FAISS needed for v1. If CPU chat latency matters later: `faiss-cpu` (MIT) IVF-PQ (a new pip
  install into the ml venv; tell the user first).
- **Retrieval quality check** before any training: recall@k on the 100 eval questions = share of questions whose top-k passages contain the
  key facts (Qwen2.5-7B grader, the factuality grader's prompt adapted to "does this passage state the key fact"). Gate: recall@3 ≥ 0.6, else
  fix chunking/embedder first.

## 3. Fine-tune format

seq 512 fits 3 passages: `You: {question}\nSources:\n[1] {≤100 tokens}\n[2] …\n[3] …\nAura: {answer}` (question ~30 + passages ~300 + answer ≤ 160).
The passages sit in the prompt part, so `finetune_chat.py`'s reply-only loss trains only the answer; no trainer change.

**Training data (RAG-ft_v2):** the ft_v2 explain/chat/oasst2 prompts (~160k) each get their top-3 passages. Qwen2.5-7B-Instruct (Apache-2.0,
via vLLM) regenerates the answer *given the passages* with the ft_v2 persona and mood rules plus: "use the sources when they answer the
question, don't quote them verbatim, don't invent facts beyond them; if they're irrelevant, answer from general knowledge and say less".
20% get 3 random (irrelevant) passages instead, so AURA learns to ignore bad retrievals. Arithmetic and story rows are kept from ft_v2 with
`Sources: none`. Same filters as ft_v2 (identity, length, LaTeX/code, near-dups), and a new one: reject answers that copy > 30 consecutive
tokens from a passage.

## 4. Evaluation (factuality eval, 100 questions, same grader, both encodings, greedy + 3 sampled, pooled seeds)

| arm | model | retrieval at eval | question |
|---|---|---|---|
| R0 | `v2base/ft/b10_s0, b10_s1` (existing scores 0.302 joined / 0.316 pieces, pooled) | none | the bar |
| R1 | the same models | top-3 passages prepended | does retrieval help a model never trained to use it? |
| R2 | RAG fine-tune from the 10B base, seeds 0 and 1 | top-3 | the method |
| R2-closed | R2 | none | did the RAG fine-tune cost closed-book ability? |
| R2-oracle | R2 | the question's reference answer as passage [1] | upper bound (how well AURA reads) |

**Rule (to pre-register before running):** retrieval counts if **R2 beats R0** (pooled seeds, paired bootstrap over questions, CI > 0) **and**
R2's false-claim rate isn't higher than R0's **and** R2-closed isn't more than 0.02 below R0. Reported: fully-correct rate, recall@3, and how
often answers copy passages.

## 5. Cost

| step | where | time |
|---|---|---|
| chunk + filter S1 (~6M passages) | CPU | ~30-45 min |
| embed 6M passages (bge-small fp16, ~15-20k passages/s) | GPU | ~10 min (+ CPU tokenization ~15 min) |
| retrieval for 160k training prompts + eval | GPU | ~5 min |
| recall@k check (grader) | GPU | ~10 min |
| teacher regeneration with passages (~160k, longer prompts, ~50 samples/s) | GPU | ~55 min + 3 min load |
| RAG fine-tune × 2 seeds (~25 min each at vanilla2 l100 speed) | GPU | ~50 min |
| factuality eval: R1, R2 ×2, R2-closed ×2, R2-oracle ×2 (generate + grade) | GPU | ~30 min |
| smoke tests of every phase | GPU | ~20 min |
| **total** | | **≈ 3.5 GPU-h + ~1 h CPU** |

**Disk:** passages text ~3 GB, embeddings 4.6 GB, RAG-ft data ~0.5 GB, 2 fine-tunes ~1 GB → **~9 GB** (~10 GB with S2).

## 6. Code needed (~half a day, CPU)

`scripts/retrieval/build_store.py` (chunk, filter, leakage check), `embed_store.py`, `cosf_mi/retrieval.py` (load, query, top-k),
`scripts/distill/build_rag_ft.py` (retrieve + teacher with passages + filters), `--retrieve k` and `--oracle` in `factuality_gen.py`, a recall@k
script, and later `--retrieve` in `chat_with_aura.py`. The queue follows the standing rules (smoke every phase, FAILED marker, telemetry).
