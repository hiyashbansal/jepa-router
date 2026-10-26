#!/usr/bin/env bash
# Next-step experiments. GPU box only (Colab T4), not a laptop. Runs one job at a time.
# Usage (from the repo root): bash colab_next_steps.sh   -> prints result blocks, writes results_next.zip
[ -f routerbench_0shot.pkl ] || wget -q https://huggingface.co/datasets/withmartian/routerbench/resolve/main/routerbench_0shot.pkl
pip -q install sentence-transformers

E="--source routerbench --path routerbench_0shot.pkl --embedder sbert:all-MiniLM-L6-v2 --epochs 60"
MIX="grade-school-math mbpp mmlu arc-challenge winogrande"
PAIR="--models gpt-4-1106-preview mistralai/mixtral-8x7b-chat"  # RouteLLM's own strong/weak pair
run() {
  name=$1; shift; echo "### $name  ($(date +%T))"
  python -m "$@" --out results/$name > results_$name.log 2>&1; echo "exit=$?"
  sed -n '/=== results/,$p' results_$name.log; grep "queries" results_$name.log | cut -c1-120
}

# 1. MBPP leave-one-task-out, 10 seeds (was 3)
run ood_mbpp_10s   jepa_router.train $E --tasks grade-school-math mbpp --holdout_task mbpp --seeds 10
# 2. harder task mix: random split, data scarcity, leave MMLU out
run mix            jepa_router.train $E --tasks $MIX --seeds 10
run mix_frac0.05   jepa_router.train $E --tasks $MIX --seeds 10 --train_frac 0.05
run mix_frac0.25   jepa_router.train $E --tasks $MIX --seeds 10 --train_frac 0.25
run mix_ood_mmlu   jepa_router.train $E --tasks $MIX --holdout_task mmlu --seeds 10
# 3. real RouteLLM BERT router vs ours, same split, GPT-4 vs Mixtral
run rl_gsm8k       jepa_router.train $E --tasks grade-school-math $PAIR --seeds 10 --routellm routellm/bert_gpt4_augmented
run rl_mix         jepa_router.train $E --tasks $MIX $PAIR --seeds 10 --routellm routellm/bert_gpt4_augmented
# 4. fine-tuned (not frozen) encoder
run ft_ood_mbpp    jepa_router.finetune --path routerbench_0shot.pkl --tasks grade-school-math mbpp --holdout_task mbpp
run ft_gsm8k       jepa_router.finetune --path routerbench_0shot.pkl --tasks grade-school-math

zip -qr results_next.zip results results_*.log && echo "done -> results_next.zip"
