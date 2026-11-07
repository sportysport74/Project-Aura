# Retrieval experiment: factuality (pre-registered rules applied)

- recall `recall_corpus.jsonl`: @1 0.56, @3 **0.81**
- recall `recall_wiki.jsonl`: @1 0.45, @3 **0.64**


## joined encoding, sampled_rep (and greedy score)

| arm | score | greedy score | fully correct | false claims | (Wikipedia) tag | copies ≥ 8 words of a source |
|---|---|---|---|---|---|---|
| R0 | **0.308** | 0.370 | 10.0% | 33.0% | 0.0% | 0.0% |
| R0 pooled (b10 s0+s1) | **0.302** | 0.360 | 9.0% | 33.7% | 0.0% | 0.0% |
| R1c | **0.172** | 0.120 | 3.0% | 11.3% | 0.0% | 2.3% |
| R2c | **0.512** | 0.502 | 23.3% | 13.2% | 0.0% | 28.3% |
| R2w | **0.440** | 0.490 | 14.3% | 19.7% | 91.8% | 12.3% |
| R2-closed | **0.302** | 0.355 | 5.3% | 32.3% | 0.0% | 0.0% |
| R2-oracle | **0.496** | 0.610 | 22.7% | 17.5% | 0.0% | 10.5% |

## pieces encoding, sampled_rep (and greedy score)

| arm | score | greedy score | fully correct | false claims | (Wikipedia) tag | copies ≥ 8 words of a source |
|---|---|---|---|---|---|---|
| R0 | **0.323** | 0.390 | 10.0% | 33.0% | 0.0% | 0.0% |
| R0 pooled (b10 s0+s1) | **0.312** | 0.378 | 8.0% | 32.3% | 0.0% | 0.0% |
| R1c | **0.242** | 0.240 | 6.0% | 18.3% | 0.0% | 6.0% |
| R2c | **0.512** | 0.557 | 19.8% | 9.5% | 0.0% | 27.5% |
| R2w | **0.463** | 0.525 | 14.0% | 16.5% | 93.0% | 18.8% |
| R2-closed | **0.287** | 0.372 | 3.8% | 34.7% | 0.0% | 0.0% |
| R2-oracle | **0.499** | 0.640 | 20.5% | 16.3% | 0.0% | 11.5% |

## Paired comparisons (joined, sampled_rep; B − A, 95% CI)

| A | B | B − A | CI | n |
|---|---|---|---|---|
| R0 | R2c | +0.2042 | [+0.1500, +0.2600] | 100 |
| R2c | R2w | -0.0725 | [-0.1225, -0.0250] | 100 |
| R0 | R2-closed | -0.0058 | [-0.0425, +0.0317] | 100 |
| R0 | R1c | -0.1367 | [-0.1950, -0.0800] | 100 |
| R0 | R2-oracle | +0.1875 | [+0.1417, +0.2350] | 100 |
| R0 pooled (b10 s0+s1) | R2c | +0.2108 | [+0.1600, +0.2633] | 100 |

## Verdicts (pre-registered)

1. **Retrieval works: YES.** R2c − R0 +0.2042 (CI lower +0.1500; needs > 0); false claims R2c 13.2% vs R0 33.0% (needs ≤ R0 + 2 pts); R2-closed − R0 -0.0058 (needs ≥ −0.02).
2. **Wikipedia store as the default chat store: NO (corpus store stays the default).** R2w − R2c -0.0725 (CI lower -0.1225; needs > 0).
