cases: 60  code: 527d8ff90d39  manifest sha256: f30ea57b853f

### ss+tt

| model | row | AC@1 | AC@3 | AC@5 | MRR | fault acc | indicator acc | grounding | cost/case | LLM s/case | wall s/case | missing | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen2.5:7b-instruct | no_critic | 30% | 45% | 45% | 0.375 | 23% | n/a | pending | $0.0000 | 37.6 | 38.0 | 0 | 0 |
| qwen2.5:7b-instruct | code_rule | 60% | 63% | 63% | 0.617 | 37% | n/a | pending | $0.0000 | 50.5 | 13.2 | 0 | 0 |
| qwen2.5:7b-instruct | hybrid | 75% | 85% | 85% | 0.792 | 37% | n/a | pending | $0.0000 | 310.1 | 272.5 | 1 | 21 |
| qwen2.5:7b-instruct | llm_only | 58% | 73% | 75% | 0.653 | 33% | n/a | pending | $0.0000 | 306.2 | 109.9 | 1 | 22 |
| qwen2.5:7b-instruct | hybrid_7b | 67% | 78% | 78% | 0.725 | 35% | n/a | pending | $0.0000 | 170.0 | 135.1 | 0 | 6 |
| qwen2.5:7b-instruct | react | 72% | 85% | 85% | 0.775 | 30% | n/a | pending | $0.0000 | 19.7 | 20.4 | 0 | 0 |
| qwen2.5:7b-instruct | retrieval_only | 7% | 27% | 27% | 0.153 | 13% | n/a | pending | $0.0000 | 6.3 | 6.4 | 0 | 0 |
| code-only | largest_onset_shift | 70% | 88% | 95% | 0.800 | n/a | n/a | n/a | n/a | n/a | n/a | 0 | 0 |

code-only = largest pre/post-onset metric shift (scripts/run_deterministic_baseline.py), no LLM.

### ss

| model | row | AC@1 | AC@3 | AC@5 | MRR | fault acc | indicator acc | grounding | cost/case | LLM s/case | wall s/case | missing | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen2.5:7b-instruct | no_critic | 43% | 60% | 60% | 0.517 | 23% | n/a | pending | $0.0000 | 36.3 | 36.5 | 0 | 0 |
| qwen2.5:7b-instruct | code_rule | 87% | 90% | 90% | 0.883 | 47% | n/a | pending | $0.0000 | 54.5 | 18.0 | 0 | 0 |
| qwen2.5:7b-instruct | hybrid | 83% | 90% | 90% | 0.856 | 37% | n/a | pending | $0.0000 | 310.0 | 272.4 | 0 | 9 |
| qwen2.5:7b-instruct | llm_only | 63% | 73% | 77% | 0.679 | 33% | n/a | pending | $0.0000 | 267.6 | 96.2 | 0 | 9 |
| qwen2.5:7b-instruct | hybrid_7b | 80% | 90% | 90% | 0.850 | 47% | n/a | pending | $0.0000 | 155.3 | 120.2 | 0 | 2 |
| qwen2.5:7b-instruct | react | 63% | 80% | 80% | 0.700 | 33% | n/a | pending | $0.0000 | 22.2 | 23.0 | 0 | 0 |
| qwen2.5:7b-instruct | retrieval_only | 13% | 37% | 37% | 0.239 | 17% | n/a | pending | $0.0000 | 5.5 | 5.6 | 0 | 0 |
| code-only | largest_onset_shift | 73% | 97% | 100% | 0.847 | n/a | n/a | n/a | n/a | n/a | n/a | 0 | 0 |

code-only = largest pre/post-onset metric shift (scripts/run_deterministic_baseline.py), no LLM.

### tt

| model | row | AC@1 | AC@3 | AC@5 | MRR | fault acc | indicator acc | grounding | cost/case | LLM s/case | wall s/case | missing | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen2.5:7b-instruct | no_critic | 17% | 30% | 30% | 0.233 | 23% | n/a | pending | $0.0000 | 38.8 | 39.6 | 0 | 0 |
| qwen2.5:7b-instruct | code_rule | 33% | 37% | 37% | 0.350 | 27% | n/a | pending | $0.0000 | 46.6 | 8.3 | 0 | 0 |
| qwen2.5:7b-instruct | hybrid | 67% | 80% | 80% | 0.728 | 37% | n/a | pending | $0.0000 | 310.2 | 272.5 | 1 | 12 |
| qwen2.5:7b-instruct | llm_only | 53% | 73% | 73% | 0.628 | 33% | n/a | pending | $0.0000 | 346.1 | 124.0 | 1 | 13 |
| qwen2.5:7b-instruct | hybrid_7b | 53% | 67% | 67% | 0.600 | 23% | n/a | pending | $0.0000 | 184.8 | 150.1 | 0 | 4 |
| qwen2.5:7b-instruct | react | 80% | 90% | 90% | 0.850 | 27% | n/a | pending | $0.0000 | 17.2 | 17.8 | 0 | 0 |
| qwen2.5:7b-instruct | retrieval_only | 0% | 17% | 17% | 0.067 | 10% | n/a | pending | $0.0000 | 7.2 | 7.3 | 0 | 0 |
| code-only | largest_onset_shift | 67% | 80% | 90% | 0.753 | n/a | n/a | n/a | n/a | n/a | n/a | 0 | 0 |

code-only = largest pre/post-onset metric shift (scripts/run_deterministic_baseline.py), no LLM.
