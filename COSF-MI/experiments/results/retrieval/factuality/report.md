# Retrieval experiment: factuality (pre-registered rules applied)

- recall `recall_corpus.jsonl`: @1 0.56, @3 **0.81**
- recall `recall_wiki.jsonl`: @1 0.45, @3 **0.64**


## joined encoding, sampled_rep (and greedy score)

| arm | score | greedy score | fully correct | false claims | (Wikipedia) tag | copies ≥ 8 words of a source |
|---|---|---|---|---|---|---|
| R0 | **0.310** | 0.375 | 10.0% | 32.0% | 0.0% | 0.0% |
| R0 pooled (b10 s0+s1) | **0.301** | 0.360 | 8.7% | 33.3% | 0.0% | 0.0% |
| R1c | **0.070** | 0.125 | 1.3% | 16.0% | 0.0% | 1.0% |
| R2c | **0.376** | 0.497 | 9.3% | 25.2% | 0.0% | 5.7% |
| R2w | **0.332** | 0.492 | 9.8% | 33.5% | 49.7% | 1.7% |
| R2-closed | **0.302** | 0.355 | 5.0% | 32.3% | 0.0% | 0.0% |
| R2-oracle | **0.262** | 0.613 | 5.3% | 36.5% | 0.0% | 0.3% |

## pieces encoding, sampled_rep (and greedy score)

| arm | score | greedy score | fully correct | false claims | (Wikipedia) tag | copies ≥ 8 words of a source |
|---|---|---|---|---|---|---|
| R0 | **0.327** | 0.395 | 10.0% | 33.0% | 0.0% | 0.0% |
| R0 pooled (b10 s0+s1) | **0.317** | 0.378 | 8.0% | 32.2% | 0.0% | 0.0% |
| R1c | **0.112** | 0.245 | 2.0% | 21.7% | 0.0% | 1.3% |
| R2c | **0.394** | 0.565 | 11.3% | 26.0% | 0.0% | 6.2% |
| R2w | **0.393** | 0.525 | 11.2% | 28.0% | 62.0% | 3.5% |
| R2-closed | **0.291** | 0.372 | 4.2% | 34.8% | 0.0% | 0.0% |
| R2-oracle | **0.323** | 0.635 | 6.8% | 32.0% | 0.0% | 0.7% |

## Paired comparisons (joined, sampled_rep; B − A, 95% CI)

| A | B | B − A | CI | n |
|---|---|---|---|---|
| R0 | R2c | +0.0658 | [+0.0183, +0.1167] | 100 |
| R2c | R2w | -0.0442 | [-0.0842, -0.0058] | 100 |
| R0 | R2-closed | -0.0075 | [-0.0450, +0.0300] | 100 |
| R0 | R1c | -0.2400 | [-0.2933, -0.1867] | 100 |
| R0 | R2-oracle | -0.0483 | [-0.0958, +0.0000] | 100 |
| R0 pooled (b10 s0+s1) | R2c | +0.0750 | [+0.0325, +0.1200] | 100 |

## Verdicts (pre-registered)

1. **Retrieval works: YES.** R2c − R0 +0.0658 (CI lower +0.0183; needs > 0); false claims R2c 25.2% vs R0 32.0% (needs ≤ R0 + 2 pts); R2-closed − R0 -0.0075 (needs ≥ −0.02).
2. **Wikipedia store as the default chat store: NO (corpus store stays the default).** R2w − R2c -0.0442 (CI lower -0.0842; needs > 0).
