"""Data layer.

Everything downstream works on one tiny schema, `RouteData`:
    prompts   : list[str]            N queries
    task      : np.ndarray[str]      N task labels (e.g. gsm8k, mbpp)
    models    : list[str]            M candidate LLMs
    score     : float[N, M]          quality of model m on query n (0..1)
    cost      : float[N, M]          dollar cost of model m on query n
    responses : list[list[str]]      responses[m][n] text (used as the JEPA target view)

Two sources: RouterBench (real, precomputed outcomes, no LLM calls needed) and a
synthetic generator (offline smoke test only, NOT a scientific result).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class RouteData:
    prompts: list
    task: np.ndarray
    models: list
    score: np.ndarray
    cost: np.ndarray
    responses: list

    def subset(self, idx):
        idx = np.asarray(idx)
        return RouteData(
            [self.prompts[i] for i in idx],
            self.task[idx],
            self.models,
            self.score[idx],
            self.cost[idx],
            [[r[i] for i in idx] for r in self.responses],
        )

    def __len__(self):
        return len(self.prompts)


# --------------------------------------------------------------------------- RouterBench
def load_routerbench(path, tasks=("grade-school-math",), models=None, max_n=None, seed=0):
    """Load RouterBench pickle (HF: withmartian/routerbench, file routerbench_0shot.pkl).

    Assumed columns (from the RouterBench repo; verify on first run with --inspect):
        prompt, eval_name, <model>, <model>|total_cost, <model>|model_response
    `tasks` filters eval_name (math = "grade-school-math" i.e. GSM8K). Pass None for all.
    """
    import pandas as pd

    df = pd.read_pickle(path)
    if tasks is not None:
        df = df[df["eval_name"].isin(tasks)]
    all_models = [c for c in df.columns if f"{c}|total_cost" in df.columns]
    models = models or all_models
    if max_n and len(df) > max_n:
        df = df.sample(max_n, random_state=seed)
    df = df.reset_index(drop=True)
    score = df[models].to_numpy(dtype=float)
    cost = np.stack([df[f"{m}|total_cost"].to_numpy(dtype=float) for m in models], 1)
    resp = [df[f"{m}|model_response"].astype(str).tolist() for m in models]
    return RouteData(df["prompt"].astype(str).tolist(), df["eval_name"].to_numpy(), models, score, cost, resp)


def inspect_routerbench(path):
    import pandas as pd

    df = pd.read_pickle(path)
    print(df.shape)
    print(list(df.columns))
    print(df["eval_name"].value_counts())


# --------------------------------------------------------------------------- synthetic
def make_synthetic(n=4000, seed=0):
    """Toy world: 3 task types, hidden difficulty, 4 models with task-specific skill + cost.

    Query text leaks task + difficulty through words ("multi-step", clause count), so a
    representation CAN learn them. Used only to check the pipeline runs end to end.
    """
    rng = np.random.default_rng(seed)
    tasks = np.array(["math", "code", "qa"])
    task_words = {
        "math": ["solve", "equation", "sum", "product", "fraction", "integer"],
        "code": ["function", "python", "array", "loop", "return", "string"],
        "qa": ["who", "which", "capital", "year", "river", "author"],
    }
    hard_words = ["multi-step", "prove", "optimize", "nested", "edge-case", "derive"]
    filler = ["please", "the", "a", "given", "number", "list", "value", "now", "then"]

    models = ["small", "medium", "large", "math-specialist"]
    cost_per = np.array([1.0, 4.0, 20.0, 5.0])
    skill = {  # model x task skill
        "math": np.array([0.25, 0.5, 0.85, 0.8]),
        "code": np.array([0.3, 0.55, 0.85, 0.3]),
        "qa": np.array([0.45, 0.65, 0.9, 0.35]),
    }
    prompts, tk, sc, co = [], [], [], []
    responses = [[] for _ in models]
    for _ in range(n):
        t = rng.choice(tasks)
        d = rng.beta(2, 2)
        n_hard = rng.binomial(3, d)
        words = (
            list(rng.choice(task_words[t], 3))
            + list(rng.choice(hard_words, n_hard))
            + list(rng.choice(filler, rng.integers(3, 8)))
        )
        rng.shuffle(words)
        prompts.append(" ".join(words))
        tk.append(t)
        p = 1 / (1 + np.exp(-8 * (skill[t] - d)))
        ok = (rng.random(len(models)) < p).astype(float)
        sc.append(ok)
        co.append(cost_per * (1 + 0.3 * rng.random(len(models))))
        for m in range(len(models)):
            if ok[m]:
                responses[m].append(f"correct {t} solution " + " ".join(rng.choice(task_words[t], 4)))
            else:
                responses[m].append("wrong " + " ".join(rng.choice(filler, 4)))
    return RouteData(prompts, np.array(tk), models, np.array(sc), np.array(co), responses)


def split(data: RouteData, test_frac=0.2, seed=0, holdout_task=None):
    """Random split, or (holdout_task=...) leave one task out entirely: an OOD test."""
    rng = np.random.default_rng(seed)
    n = len(data)
    if holdout_task is not None:
        te = np.where(data.task == holdout_task)[0]
        tr = np.where(data.task != holdout_task)[0]
        return data.subset(tr), data.subset(te)
    perm = rng.permutation(n)
    k = int(n * (1 - test_frac))
    return data.subset(perm[:k]), data.subset(perm[k:])
