# Gate data: hand review (2026-11-07, before any signal or threshold was computed)

- **20 random items per kind read** (`data/retrieval/gate/*.jsonl`, built by `scripts/retrieval/build_gate_sets.py`; build.json there).
- **Dropped generator (pre-registered rule): unanswerable / realtime.** It produced wrong-kind, answerable items ("What's the elevation of Mount
  Everest?", "What's the population of São Tomé and Príncipe as of 2023?", "What's the fastest way to travel from San Francisco to Los Angeles?") and
  one broken string. All 80 realtime items are excluded; unanswerable = personal 63 + niche 43 + recent 20 = 126 (split as built). The "recent"
  items are low-diversity (World Cup 2026 / Nobel 2027 rephrasings) but the right kind.
- chit-chat, garbled and on-topic: right kind in the samples. On-topic is the ft_v2 val prompts as they are (some odd ones like "introduce yourself").
- **Multi-turn: 38 of 60 kept** (all 60 read). Dropped 22: m008, m012, m013, m016, m020, m022, m025, m029, m030, m033, m034, m036, m039, m040, m044, m045, m046, m047, m048, m050, m054, m058. Reasons: follow-up doesn't make sense after turn 1
  ("What is water?" → "Why do they do that?"), reference wrong or doesn't answer the follow-up (shadows "light travels a shorter distance"; hibernation
  defined instead of explained), false premise in turn 1 (snow only in winter), duplicates (allergies, levers), off-domain (math, budgets, studying).
  Not padded to 40.
