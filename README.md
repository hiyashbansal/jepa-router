# JEPA-based LLM Router

Question: can a JEPA-style latent representation of a query route to LLMs better than a BERT-style classifier (RouteLLM)?

**Short answer so far:** mostly no. On GSM8K and on an unseen task (MBPP), JEPA matches a plain classifier head. On a harder 5-task mix, the 2-stage JEPA router wins on all 10 seeds, but only by +0.008 AUC, and a control is still needed to show the gain comes from JEPA and not from its linear-probe training. Details below.

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

## Routers compared

| Router | What it is | Where |
|---|---|---|
| raw-head | MLP on the frozen query embedding, BCE only (BERT-classifier-style baseline) | `train.py` |
| jepa-joint | same MLP, JEPA loss + BCE trained together | `train.py` |
| jepa-2stage | JEPA loss only, then freeze and fit a linear head | `train.py` |
| routellm-bert | RouteLLM's released BERT router (`routellm/bert_gpt4_augmented`), threshold sweep | `train.py --routellm` |
| ft-raw / ft-jepa | encoder itself fine-tuned, BCE vs BCE + JEPA | `finetune.py` |
| random-mix, best-single, oracle | reference points (oracle = cheapest correct model) | |

Data: [RouterBench](https://huggingface.co/datasets/withmartian/routerbench) `routerbench_0shot.pkl`: 36,497 prompts with precomputed scores, costs and responses of 11 LLMs, so no LLM calls are needed.
Embedding: frozen `sbert:all-MiniLM-L6-v2` unless stated. Split: fixed 80/20 random, or leave one task out.
Metric: **AUC**, the mean quality of the cost–quality Pareto frontier over a shared cost range (higher = better). Also reported: the cost to reach 95% of the best single model's quality. That one is averaged only over the seeds that reach the target (`cost@95%_reached`), so compare routers by AUC.

## Results

### 1. GSM8K (7,450 queries, 1,490 test)

| Router | AUC | cost to 95% of GPT-4 quality |
|---|---|---|
| random-mix | 0.5378 | $0.0077 |
| raw-head | 0.6417 ± 0.0004 | $0.0017 |
| jepa-joint | 0.6416 ± 0.0001 | $0.0017 |
| jepa-2stage | 0.6415 ± 0.0001 | $0.0020 |
| best single (gpt-4-1106-preview) | | $0.0085, quality 0.661 |
| oracle | | $0.0004, quality 0.682 |

3 seeds. All learned routers match GPT-4 quality at ~1/5 of its cost; JEPA adds nothing.

### 2. GSM8K with scarce training data (`--train_frac`, 10 seeds)
Each seed draws a different training subset, shared by all three routers.

| Train data | raw-head | jepa-joint | jepa-2stage |
|---|---|---|---|
| 5% (~300 queries) | 0.6402 ± 0.0021 | 0.6398 ± 0.0029 | 0.6397 ± 0.0030 |
| 10% | 0.6412 ± 0.0014 | 0.6404 ± 0.0011 | 0.6409 ± 0.0009 |
| 25% | 0.6415 ± 0.0006 | 0.6415 ± 0.0005 | 0.6416 ± 0.0004 |

Flat: even 5% of the data saturates every router. GSM8K has almost no routing headroom (oracle 0.682 vs GPT-4 0.661), so it can't separate methods.

### 3. Unseen task: train GSM8K, test MBPP (427 test queries, 10 seeds)

| Router | AUC | seeds reaching 95% of GPT-4 |
|---|---|---|
| random-mix | 0.5152 | |
| raw-head | 0.6412 ± 0.0079 | 4/10 |
| jepa-joint | 0.6433 ± 0.0116 | 5/10 |
| jepa-2stage | 0.6300 ± 0.0314 | 4/10 |
| GPT-4 / oracle quality | 0.686 / 0.867 | |

Paired by seed against raw-head: jepa-joint +0.0021 (wins 8/10, sign test p = 0.11), jepa-2stage −0.0113 (wins 4/10). **No significant difference.** jepa-2stage is unstable on unseen tasks (AUC 0.57–0.67). An earlier 3-seed run (`results/ood_mbpp_3s`) suggested JEPA was slightly worse; with 10 seeds that turned out to be noise.

### 4. Harder task mix: GSM8K + MBPP + MMLU + ARC-Challenge + Winogrande (24,656 queries, 10 seeds)

| Router | AUC | seeds reaching 95% |
|---|---|---|
| random-mix | 0.5457 | |
| raw-head | 0.7377 ± 0.0047 | 10/10 |
| jepa-joint | 0.7405 ± 0.0034 | 10/10 |
| **jepa-2stage** | **0.7456 ± 0.0011** | 10/10 |
| GPT-4 / oracle quality | 0.771 / 0.875 | |

Paired by seed against raw-head: **jepa-2stage +0.0079, wins 10/10 (sign test p = 0.002)**, and it is ~4× more stable; jepa-joint +0.0028, wins 7/10 (p = 0.34).
This is the first measurable JEPA gain, but it is small (~1% relative). It is also confounded: jepa-2stage ends with a frozen encoder and a linear head, a strong regulariser, while raw-head trains a full MLP for 60 epochs and may overfit. A linear-probe control is the next experiment.

### 5. Pending (in `colab_next_steps.sh`, not yet finished)
- Task mix with 5% and 25% of training data
- Task mix with MMLU held out
- RouteLLM's BERT router vs ours, GPT-4 vs Mixtral, on GSM8K and on the mix
- Fine-tuned encoder (ft-raw vs ft-jepa) on GSM8K and on the MBPP holdout

## Caveats
- Seed variance covers model init and data order only. The test split is fixed, so split variance is not measured.
- On synthetic data (`--source synthetic`) all routers score about the same, because the toy text leaks everything. It is a pipeline check, not evidence.
- `raw-head` is a stand-in for RouteLLM's classifier. The real RouteLLM BERT router is wired in (`--routellm`) but its results are still pending.

## Folder layout

```
jepa_router/        code: data.py, embed.py, model.py, routing.py, train.py (frozen encoder), finetune.py
colab_next_steps.sh all pending GPU experiments, run one at a time
results/<run>/      results.json (+ curves) and run.log, or output.txt pasted back from Colab
requirements.txt
local only, not in git: routerbench_0shot.pkl, .venv/, jepa_router_colab.zip, jepa-router-original.zip (the v0 download)
```

## Run

Heavy jobs (anything on RouterBench with sbert, `--routellm` or `finetune.py`) belong on a GPU machine (e.g. Colab T4), one at a time. On a laptop they max out the CPU. Only the synthetic smoke test is laptop-friendly.

```bash
pip install -r requirements.txt

# laptop-safe smoke test (synthetic data, seconds)
OMP_NUM_THREADS=1 python -m jepa_router.train --source synthetic --embedder tfidf --epochs 5

# data: https://huggingface.co/datasets/withmartian/routerbench  ->  routerbench_0shot.pkl
python -m jepa_router.train --source routerbench --path routerbench_0shot.pkl --inspect
python -m jepa_router.train --source routerbench --path routerbench_0shot.pkl \
    --tasks grade-school-math --embedder sbert:all-MiniLM-L6-v2 --epochs 60
#   --tasks matches eval_name by prefix (mmlu = all mmlu-* subjects)
#   --holdout_task mbpp     leave one task out (OOD)
#   --train_frac 0.05       per-seed training subset (data scarcity)
#   --seeds 10              number of seeds
#   --models A B --routellm routellm/bert_gpt4_augmented   add RouteLLM's BERT router
python -m jepa_router.finetune --path routerbench_0shot.pkl --tasks grade-school-math   # fine-tuned encoder
```

Colab: `zip -r jepa_router_colab.zip jepa_router colab_next_steps.sh`, upload it, then
`!unzip -qo jepa_router_colab.zip && bash colab_next_steps.sh`. It downloads the data, runs every pending job in turn and writes `results_next.zip`. If Colab already has an older zip, delete it first: uploads get renamed to `... (1).zip`.

## Next experiments
1. Linear-probe control for result 4: fit raw-head the jepa-2stage way (random-init or BCE-trained encoder, frozen, linear head). If the gap survives, it is JEPA.
2. Finish the pending runs above, especially RouteLLM vs ours on the same split.
3. Add split variance (several random splits), not just seed variance.
4. Ablate the target view: best response vs per-model response vs problem solution text.
5. Add latency to the cost term.
