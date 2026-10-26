# JEPA-based LLM Router (v0 prototype)

Question: can a JEPA-style latent representation of a query route to LLMs better than a BERT-style classifier (RouteLLM)?

## Idea in one picture

```
query x ──► context encoder ──► z_x ──► predictor ──► predicted z_y
                                 │                         ▲ (match in latent space)
                                 ▼                         │
                         head: P(success) per LLM     target encoder (EMA) ◄── y = best known solution
                                 │
                 pick argmax  P(success) − λ · cost
```

- Views: x = the query, y = the best model's response to it (best score, ties → cheapest).
- To predict what a good solution looks like, z_x has to encode task type and difficulty. That is the "world model of the task".
- Loss (joint) = latent cosine loss + variance regulariser (anti-collapse) + BCE on per-model success.
- Frozen text embedder underneath (v0). End-to-end fine-tuning is a later step.

## Run

```bash
pip install -r requirements.txt

# offline smoke test (synthetic data, only checks the pipeline)
python -m jepa_router.train --source synthetic --embedder tfidf

# real: RouterBench math (GSM8K). Download routerbench_0shot.pkl from
# https://huggingface.co/datasets/withmartian/routerbench
python -m jepa_router.train --source routerbench --path routerbench_0shot.pkl --inspect   # check columns first
python -m jepa_router.train --source routerbench --path routerbench_0shot.pkl \
    --tasks grade-school-math --embedder sbert:all-MiniLM-L6-v2 --epochs 60
python -m jepa_router.train ... --tasks grade-school-math mbpp --holdout_task mbpp          # OOD test
```

## Routers compared (same frozen embedding, same head size)
random-mix, raw-head (BERT-classifier-style baseline), jepa-joint, jepa-2stage, plus the oracle.
Metrics: area under the cost–quality frontier on a shared cost range, and cost needed to reach 95% of the best single model's quality.

## Status / honest caveats
- Pipeline runs end to end on synthetic data. There, all learned routers score about the same (~0.78 AUC vs 0.62 random): that is expected because the toy text leaks everything. **It is not evidence for or against JEPA.**
- RouterBench column names in `data.py` verified with `--inspect` on `routerbench_0shot.pkl` (36,497 rows; GSM8K = 7,450).
- The real RouteLLM (matrix factorization / BERT trained on Arena preference data) is not reproduced here; `raw-head` is a stand-in. Next step: run their repo on the same split.

## Results: RouterBench GSM8K
Frozen `sbert:all-MiniLM-L6-v2`, 60 epochs, 11 models, fixed 80/20 split (1,490 test queries). AUC = area under cost–quality frontier.

| Router | AUC | cost to 95% of best quality |
|---|---|---|
| random-mix | 0.5378 | $0.0077 |
| raw-head | 0.6417 ± 0.0004 | $0.0017 |
| jepa-joint | 0.6416 ± 0.0001 | $0.0017 |
| jepa-2stage | 0.6415 ± 0.0001 | $0.0020 |
| best single (gpt-4-1106-preview) | — | $0.0085 (quality 0.661) |
| oracle | — | $0.0004 (quality 0.682) |

Data scarcity (`--train_frac`, 10 seeds, each seed a different training subset shared by all routers):

| Train data | raw-head | jepa-joint | jepa-2stage |
|---|---|---|---|
| 5% (~300 queries) | 0.6402 ± 0.0021 | 0.6398 ± 0.0029 | 0.6397 ± 0.0030 |
| 10% | 0.6412 ± 0.0014 | 0.6404 ± 0.0011 | 0.6409 ± 0.0009 |
| 25% | 0.6415 ± 0.0006 | 0.6415 ± 0.0005 | 0.6416 ± 0.0004 |

Takeaways:
- Learned routers match GPT-4 quality at ~1/5 of its cost, but **JEPA gives no gain over the plain head**, at full data or with as little as 5%.
- GSM8K leaves little headroom: the oracle (0.682) is barely above GPT-4 (0.661), and even 5% of the data saturates every router. A task mix with more model disagreement is a better testbed.

## Results: leave-one-task-out (train GSM8K, test MBPP)
`--tasks grade-school-math mbpp --holdout_task mbpp`, same setup, 3 seeds, 427 MBPP test queries (run on Colab).

| Router | AUC | cost to 95% of best quality* |
|---|---|---|
| random-mix | 0.5152 | $0.0084 |
| raw-head | **0.6477 ± 0.0071** | $0.0045 |
| jepa-joint | 0.6415 ± 0.0173 | $0.0003 |
| jepa-2stage | 0.6289 ± 0.0362 | $0.0003 |
| best single (gpt-4-1106-preview) | — | $0.0094 (quality 0.686) |
| oracle | — | $0.0005 (quality 0.867) |

\*Mean over seeds that reach the target only (`nanmean`), so it is not comparable across rows; use AUC.

- Out of distribution, **JEPA is slightly worse than the plain head and 2–5× less stable across seeds**. With 3 seeds the gap is within noise.
- Unlike GSM8K, MBPP has real headroom (oracle 0.867 vs GPT-4 0.686), so it is the better place to test routing ideas.

## Next experiments
1. More seeds (10) on the MBPP holdout; add MMLU to the task mix.
2. Ablate the target view: best response vs per-model response vs problem solution text.
3. Add latency to the cost term; add a fine-tuned (not frozen) encoder.
4. Repeat the sample-efficiency curve on a harder task mix (on GSM8K it is flat, see above).
