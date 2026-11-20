# Held-out paired tests (AC@1, 60 cases)

Exact two-sided McNemar test on per-case AC@1 (`docs/results/heldout_qwen_cases.csv`). B better / B worse = cases where only B / only A ranks the true service first. Not adjusted for multiple comparisons.

| A | B | what it tests | AC@1 A | AC@1 B | B better | B worse | p |
|---|---|---|---:|---:|---:|---:|---:|
| no_critic | hybrid | critic vs none | 18/60 | 45/60 | 28 | 1 | 1.1e-07 |
| no_critic | code_rule | code-rule critic vs none | 18/60 | 36/60 | 18 | 0 | 7.6e-06 |
| no_critic | llm_only | LLM critic vs none | 18/60 | 35/60 | 18 | 1 | 7.6e-05 |
| llm_only | hybrid | hybrid vs LLM-only critic | 35/60 | 45/60 | 11 | 1 | 0.0063 |
| code_rule | hybrid | hybrid vs code-rule critic | 36/60 | 45/60 | 11 | 2 | 0.022 |
| hybrid_7b | hybrid | 14B vs 7B critic | 40/60 | 45/60 | 12 | 7 | 0.36 |
| react | hybrid | TriageRCA hybrid vs ReAct | 43/60 | 45/60 | 10 | 8 | 0.81 |
| largest_onset_shift | hybrid | hybrid vs code-only | 42/60 | 45/60 | 8 | 5 | 0.58 |
| largest_onset_shift | react | ReAct vs code-only | 42/60 | 43/60 | 8 | 7 | 1 |
