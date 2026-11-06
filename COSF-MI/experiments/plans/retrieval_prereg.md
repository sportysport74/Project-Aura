# Pre-registered: retrieval for AURA (written 2026-11-06, before any retrieval code ran; plan: `experiments/plans/retrieval.md`)

User decisions (2026-11-06): the plan is approved, with **Simple English Wikipedia as an inference-only store**. The retrieval fine-tune uses
passages from **our own corpus store only**, so the weights never train on CC BY-SA text. **When an answer uses a Wikipedia passage, "(Wikipedia)" is
shown after it.** Compared on the factuality eval: b10_s1 vs retrieval with the corpus store vs retrieval with the Wikipedia store.

## Stores

- **corpus**: Cosmopedia v2 + FineWeb-Edu (int_score ≥ 3), the shards in `data/corpus_v1/raw` (ODC-By), cut into passages of ≤ ~90 words on
  sentence boundaries; capped at 6M passages (uniform subsample if over).
- **wiki**: Simple English Wikipedia `wikimedia/wikipedia` `20231101.simple` (CC BY-SA 3.0 + GFDL), cut the same way, with the article title kept.
  **Inference only**: never in training data, and the store isn't redistributed.
- Leakage filter on both: drop any passage that contains a word 8-gram of a factuality reference answer.
- Embedder `BAAI/bge-small-en-v1.5` (MIT): CLS, L2-normalized, its query instruction for questions. Top-k by inner product.

## Arms (factuality eval: the 100 questions, Qwen2.5-7B grader, both encodings, greedy + 3 sampled seeds)

| arm | model | retrieval at eval |
|---|---|---|
| **R0** | `v2base/ft/b10_s1` | none (bar; also reported: b10_s0+s1 pooled) |
| R1c | `v2base/ft/b10_s1` | corpus top-3, prompt format below (never trained with sources) |
| **R2c** | RAG fine-tune from the 10B base, seeds 0 and 1 (pooled) | **corpus** top-3 |
| **R2w** | the same RAG models | **wiki** top-3 |
| R2-closed | the same RAG models | none |
| R2-oracle | the same RAG models | the question's reference answer as source [1] (upper bound) |

Prompt with sources: `{question}\n\nSources:\n[1] {passage}\n[2] …\n[3] …` (each passage ≤ ~100 AURA tokens; whole conversation ≤ 512).

**"(Wikipedia)" attribution:** an answer counts as using a Wikipedia passage if ≥ 3 distinct content words (≥ 4 letters, not stopwords, not
in the question) of the answer appear in a retrieved Wikipedia passage. Then " (Wikipedia)" is appended to the shown answer. **The grader sees
the answer without the tag.** The share of tagged answers is reported.

## Gate (before any training)

Retrieval recall@3 on the 100 questions (share of questions with ≥ 1 of the top-3 passages stating the key fact, Qwen grader): **corpus ≥ 0.6**,
else stop and report (chunking/embedder first). Wiki recall is reported but doesn't gate (it's inference-only).

## Decision rules (joined encoding, sampled_rep; paired bootstrap over the 100 questions, 10k draws; RAG seeds pooled)

1. **Retrieval works** if **R2c beats R0** (95% CI > 0) **and** R2c's false-claim rate is not higher than R0's by more than 0.02 (absolute)
   **and** R2-closed is not more than 0.02 below R0 (no closed-book regression from the RAG fine-tune).
2. **Wikipedia store vs corpus store:** the Wikipedia store becomes the default chat store only if **R2w beats R2c** (CI > 0); otherwise the
   corpus store is the default.
3. Reported, deciding nothing: R1c vs R0 (retrieval without training), R2-oracle (how well AURA reads a perfect source), the pieces encoding,
   fully-correct rates, recall@1/3 for both stores, the "(Wikipedia)" tag rate, and how often answers copy ≥ 8 consecutive words of a passage.

## RAG fine-tune data

ft_v2 explain/chat/oasst2 prompts (train + val) each get the corpus top-3. **20% get 3 random corpus passages instead** (to learn to ignore
irrelevant sources). Qwen2.5-7B-Instruct (Apache-2.0, vLLM) rewrites the answer given the sources, with the ft_v2 persona and mood rules plus:
use the sources when they answer the question; don't copy them; don't add facts beyond them and general knowledge; if they're irrelevant, answer
briefly from general knowledge. Same filters as ft_v2, plus: reject a reply that copies ≥ 12 consecutive words from a source. Arithmetic, story and
gsm8k rows are kept from ft_v2 unchanged. Fine-tune: `finetune_chat.py --pilot data/distill_pilot/ft_rag`, seq 512, lr 1e-4, 1 epoch, seeds 0/1,
from `checkpoints/v2base/main/best_model.pt`.

## Safety

Every phase smoke-tested first (tiny stores, ~200 teacher requests, 20-step fine-tunes, `--limit 2` evals). Any stop writes `run_control/FAILED`
and sends a phone alert (ntfy); the chain's end sends one too. No auto-resume.
