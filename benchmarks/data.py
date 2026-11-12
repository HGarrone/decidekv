"""Public datasets used by the benchmarks. Downloaded on first use into ./data (ignored by git)."""
import json
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")


def _wordnet():
    import nltk
    try:
        from nltk.corpus import wordnet as wn
        wn.ensure_loaded()
    except LookupError:
        nltk.download("wordnet", quiet=True)
        from nltk.corpus import wordnet as wn
    return wn


def wordnet(n_options: int, n_queries: int = 150, core: int = 200, seed: int = 0):
    """Reverse dictionary on WordNet nouns (definition -> word).

    Options: one entry per distinct noun, most frequent synsets first, the first `n_options` of them, shuffled
    with a seed that depends on the size. Queries: `n_queries` definitions of synsets among the first `core`
    entries, identical for every size. A prediction is correct when it is any lemma of the gold synset."""
    wn = _wordnet()
    seen, pool = set(), []
    for s in sorted(wn.all_synsets("n"), key=lambda s: -sum(lemma.count() for lemma in s.lemmas())):
        w = s.lemmas()[0].name().replace("_", " ")
        if w.lower() in seen or not w.isascii():
            continue
        seen.add(w.lower())
        pool.append((w, s))
        if len(pool) >= n_options:
            break
    idx = random.Random(seed).sample(range(min(core, len(pool))), n_queries)
    queries = [(pool[i][1].definition(), {lemma.name().replace("_", " ") for lemma in pool[i][1].lemmas()})
               for i in idx]
    options = [w for w, _ in pool]
    random.Random(n_options).shuffle(options)
    return options, queries


def correct(answer: str, gold: set) -> bool:
    a = answer.strip().strip('"').lower()
    return any(a == g.lower() for g in gold)


def load_json(name):
    p = os.path.join(RESULTS, name)
    return json.load(open(p)) if os.path.exists(p) else {}


def save_json(name, obj):
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(obj, open(os.path.join(RESULTS, name), "w"), indent=2)


# ---------------------------------------------------------------- downloads
def _fetch(url: str, path: str) -> str:
    if not os.path.exists(path):
        import urllib.request
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


BANKING_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/"
TYPED_URL = "https://huggingface.co/datasets/LocalLLaMA/typed-decisions/resolve/main/all/"


def banking77(n_queries: int = 300, seed: int = 0):
    """Banking77 intent classification (PolyAI): 77 intents, random test subset."""
    import csv
    cats = json.load(open(_fetch(BANKING_URL + "categories.json", os.path.join(DATA, "banking77", "categories.json"))))
    rows = list(csv.DictReader(open(_fetch(BANKING_URL + "test.csv", os.path.join(DATA, "banking77", "test.csv")),
                                    encoding="utf8")))
    random.Random(seed).shuffle(rows)
    options = [c.replace("_", " ") for c in cats]
    return options, [(r["text"], {r["category"].replace("_", " ")}) for r in rows[:n_queries]]


def banking77_train():
    import csv
    path = _fetch(BANKING_URL + "train.csv", os.path.join(DATA, "banking77", "train.csv"))
    return [(r["text"], r["category"].replace("_", " ")) for r in csv.DictReader(open(path, encoding="utf8"))]


def wordnet_train(exclude_definitions: set, n: int = 12000, seed: int = 0):
    """(definition, word) pairs from frequent noun synsets, excluding the test definitions."""
    wn = _wordnet()
    syns = sorted(wn.all_synsets("n"), key=lambda s: -sum(lemma.count() for lemma in s.lemmas()))[:40000]
    pairs = [(s.definition(), s.lemmas()[0].name().replace("_", " ")) for s in syns
             if s.definition() not in exclude_definitions and s.lemmas()[0].name().isascii()]
    random.Random(seed).shuffle(pairs)
    return pairs[:n]


def typed_decisions(split: str = "test"):
    """LocalLLaMA/typed-decisions (Apache-2.0): agent traces with five typed questions each."""
    import pandas as pd
    path = _fetch(TYPED_URL + f"{split}-00000-of-00001.parquet",
                  os.path.join(DATA, "typed_decisions", f"{split}.parquet"))
    return [dict(id=r["id"], workflow=r["workflow"], state=json.loads(r["state"]),
                 questions=json.loads(r["questions"]), gold=json.loads(r["gold"]))
            for _, r in pd.read_parquet(path).iterrows()]


def gold_key(g: dict) -> str:
    """Most probable option of a gold distribution (the official scoring rule)."""
    p = g["probabilities"]
    return max(p, key=p.get)


def to_text(x, indent: int = 0) -> str:
    """Plain-text rendering of a structured state (the format CLM's heads are trained on); used for every method."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, (int, float)):
        return str(x)
    pad = " " * indent
    if isinstance(x, dict):
        parts = [f"{pad}{k}:\n{to_text(v, indent + 2)}" if isinstance(v, (dict, list)) and v else f"{pad}{k}: {to_text(v)}"
                 for k, v in x.items()]
        return ("\n\n" if indent == 0 else "\n").join(parts)
    if isinstance(x, (list, tuple)):
        return "\n".join(f"{pad}-\n{to_text(v, indent + 2)}" if isinstance(v, (dict, list)) and v else f"{pad}- {to_text(v)}"
                         for v in x)
    return json.dumps(x, ensure_ascii=False)
