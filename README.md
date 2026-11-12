# kvindex

**Fast decisions over many options with a local LLM.**

`kvindex` picks one option out of a list for a given state (a message, a definition, the state of a game) with
a local LLM. It is a tool, not a new method: it packages known techniques so that one call does the right thing
at every list size.

- **Up to 3,000 options** the whole list is written once into the model's KV cache. Each decision then computes
  only the tokens of the new state, and the answer is decoded so that it can only be one of the options.
- **Above 3,000 options** an embedding catalogue selects the 50 options closest to the state and the LLM chooses
  among them.
- **Optionally**, the catalogue can be trained on examples of your own task in about a minute. Only a small
  projection of the embedding vectors is trained; the LLM is never trained.

It speaks the `Choice` / `Score` / `Noul` question types of [OpenJev](https://github.com/GPT-AGI/OpenJev), runs on
any GGUF model that [Ollama](https://ollama.com) has downloaded, and works with any embedding model.

```python
from kvindex import KV

kv = KV("qwen3:8b", embed="qwen3-embedding:4b")
intents = kv.index(["card arrival", "lost or stolen card", "refund not showing up"],
                   noun="intent", state_label="Customer message",
                   question="Which intent does this customer message express?")
intents.choose("I ordered my card two weeks ago and it's still not here").choice
# 'card arrival'
```

## Install

Download the repository (green **Code** button, **Download ZIP**, or `git clone`) and install it from its folder:

```bash
git clone https://github.com/HGarrone/kvindex
cd kvindex
pip install -e ".[llamacpp]"         # the library, llama-cpp-python, numpy
ollama pull qwen3:8b                 # any pure-attention GGUF model
ollama pull qwen3-embedding:4b       # any embedding model; only needed above 3,000 options or for training
```

On Windows, if loading the model crashes with "illegal instruction", the installed `llama-cpp-python` wheel was
built for a CPU feature the machine lacks. This command downloads the matching official llama.cpp build and
checks its SHA-256:

```bash
python -m kvindex.get_llamacpp
```

Run the offline tests with `pip install pytest` and `pytest`.

## Usage

### One list of options, many states

```python
from kvindex import KV

kv = KV("qwen3:8b", embed="qwen3-embedding:4b")
tools = kv.index({"search_flights": "find flights between two cities",
                  "book_hotel": "reserve a hotel room",
                  "cancel_booking": "cancel an existing reservation"},
                 noun="tool", state_label="User request", question="Which tool should be called?")

answer = tools.choose("I need a room in Rome from Friday to Sunday")
answer.choice        # 'book_hotel'
answer.confidence    # probability of the decoded answer
```

Options are a list of names or a `{name: description}` dictionary. `noun`, `state_label` and `question` are
written into the prompt: naming things for the model helps.

`kv.index` returns the right kind of index for the size of the list. Up to `flat_max` options (3,000 by default,
`KV(..., flat_max=...)`) the list lives in the KV cache; above it the embedding catalogue is used and `k` (50 by
default) options are shown to the LLM.

### Several typed questions about one state

```python
from kvindex import KV, Choice, Score, Noul

r = kv.decide(
    state={"customer": "This is the third time I write! My card was stolen yesterday."},
    questions={
        "intent": Choice(instructions="What is the customer's request?", options=["lost or stolen card", "refund"]),
        "mood": Score(instructions="How upset is the customer?", criteria=["calm", "annoyed", "furious"]),
        "escalate": Noul(instructions="Should this be escalated to a human right now?"),
    },
)
r.answers["intent"].choice, r.answers["mood"].score, r.answers["escalate"].noul
```

### Trial and error

`exclude=` removes options from the decoding, so an attempt that was already refuted cannot be chosen again:

```python
tried = []
while True:
    cell = grid.choose(game_state, exclude=tried).choice
    if shoot(cell):
        break
    tried.append(cell)
```

### Training the catalogue on your own examples

When you have examples of the form (state, right option) for a fixed list of options, the catalogue can learn
from them. Training is optional and is switched off unless you call it.

```python
examples = [("I still haven't received my new card", "card arrival"), ...]      # a few thousand pairs

head = kv.train(options, examples, question="Which intent does this customer message express?",
                out="intents_head.npz")
head.report          # validation accuracy before and after, epochs, seconds

intents = kv.index(options, head=head, question="Which intent does this customer message express?")
intents.choose("where is my card?").choice
```

With a trained head the catalogue decides alone (no LLM call) unless you pass `k` greater than 1. A head belongs
to the embedding model and to the list of options it was trained with. It lowers the accuracy on options that had
no training example (see the results), so train with examples for every option you care about.

### Any embedding model

`embed` is the name of an embedding model downloaded by Ollama, or any function that maps a list of texts to
L2-normalised vectors. The text sent to the embedding model for a state is `query_template`:

```python
kv = KV("qwen3:8b", embed="bge-m3", query_template="{state}")
kv = KV("qwen3:8b", embed=my_function)          # my_function(list_of_texts) -> array of unit vectors
```

The default template, `"Instruct: {question}\nQuery:{state}"`, suits instruction-tuned embedding models such as
Qwen3-Embedding.

### Engines

`KV(model)` runs llama.cpp in the Python process and reads the GGUF file from Ollama's model folder (or a `.gguf`
path). `n_ctx` (16,384 tokens by default) must hold the option list; `kv_type="q8_0"` halves the memory of the
cache. `KV(model, backend="ollama")` talks to the Ollama server instead: it gives the same kind of answers, but the
confidence of a choice is not available there (it is reported as 0) and every number below was measured with the
llama.cpp engine. Call `kv.close()` to free the GPU.

## How it works

```
up to flat_max options                         above flat_max, or with a trained head

[ instructions + all options ]  cached once    embedding catalogue -> the k closest options
[ state ]                       computed       [ instructions + k options ] [ state ] [ question ]
[ question ] -> answer          constrained    -> answer, constrained to the k options
```

In a causal LLM the keys and values of a token depend only on the tokens before it. With the options first and
unchanged, their KV cache is computed once and reused exactly by every later call; the state's tokens then attend
to every token of every option. The answer is decoded through a trie of the option names, so it is always one of
the options. Above a few thousand options the list no longer pays for itself in the cache, and the standard
retrieve-then-rerank pipeline takes over.

## Results

Same machine (RTX 4070 SUPER, 12 GB), same model (Qwen3-8B, llama.cpp, KV cache `q8_0`), same queries and the same
prompt blocks for every method measured here. Every cell is accuracy · median time per decision. The tables are
generated from the files in `results/` by `python benchmarks/report.py`.

### Against other tools

<!-- table:comparison -->
| Method | Training | Banking77 (77 intents) | WordNet (200 words) | Typed Decisions | Aim game (games won, time per shot) |
|---|---|---|---|---|---|
| **kvindex** | none | 60.3% · 66 ms | 72.7% · 38 ms | 51.3% · 72 ms | 95.0% · 127 ms |
| **kvindex, trained catalogue** | on the task | 93.0% · 15 ms | – | – | – |
| OpenJev architecture, shared prompt | none | 39.7% · 90 ms | 63.3% · 132 ms | 59.3% · 67 ms | 25.0% · 423 ms |
| OpenJev library, own prompt, Ollama | none | 44.0% · 117 ms | 22.7% · 164 ms | 58.1% · 62 ms | 25.0% · 469 ms |
| CLM-8B as released | none | 4.3% · 35 ms | 12.7% · 21 ms | 36.8% · 79 ms | 15.0% · 124 ms |
| CLM-8B, heads fine-tuned | on the task | 84.3% · 21 ms | 26.0% · 21 ms | 71.0% · 79 ms | – |
| Embedding model alone | none | 65.7% · 16 ms | – | – | – |
| Embedding top-50, then LLM | none | 56.3% · 127 ms | – | – | – |
| State-first prompt, constrained decoding | none | 61.0% · 138 ms | – | – | – |
| Cross-encoder, one LLM call per option (30 messages) | none | 70.0% · 1,589 ms | – | – | – |
<!-- /table:comparison -->

- *OpenJev architecture*: OpenJev's design (state first, options after it, answer read from the logits of the
  first token of each option) re-implemented in `benchmarks/baselines.py` on the shared prompt and engine. It is
  not the OpenJev library.
- *OpenJev library* and both *CLM-8B* rows come from `results/reference_qwen3_8b.json`: earlier measurements of
  this project on the same machine and model weights, made with scripts that are not part of the repository.
  CLM-8B is Qwen3-8B plus the released CLM-v0.1-8B heads.
- Banking77: 300 test messages. WordNet: 150 definitions, 200 candidate words. Typed Decisions: 2,000 decisions.
  Aim game: 20 games on a 15×15 grid, at most 25 shots.

### Growing lists

WordNet reverse dictionary (definition → word), 600 definitions:

<!-- table:scale -->
| Options | Index chosen by `KV.index` | Accuracy | Time per decision | Index build | Right answer among the 50 candidates |
|---|---|---|---|---|---|
| 1,000 | Index | 66.7% | 43 ms | 1 s | – |
| 3,000 | Index | 56.3% | 51 ms | 2 s | – |
| 10,000 | TopKIndex | 54.0% | 96 ms | 13 s | 89.5% |
| 30,000 | TopKIndex | 51.5% | 105 ms | 34 s | 86.5% |
<!-- /table:scale -->

The same task with 150 definitions, next to the tools that were measured on it:

<!-- table:scale150 -->
| Options | kvindex | OpenJev architecture | CLM-8B as released | CLM-8B, heads fine-tuned |
|---|---|---|---|---|
| 200 | 72.7% · 40 ms | 63.3% · 132 ms | 12.7% · 21 ms | 26.0% · 21 ms |
| 1,000 | 62.7% · 42 ms | not supported (max 255) | 4.7% · 21 ms | 16.7% · 21 ms |
| 3,000 | 57.3% · 50 ms | not supported (max 255) | 4.0% · 21 ms | 10.7% · 21 ms |
| 10,000 | 50.7% · 94 ms | not supported (max 255) | 3.3% · 21 ms | 8.7% · 21 ms |
| 30,000 | 46.0% · 103 ms | not supported (max 255) | not measured | not measured |
<!-- /table:scale150 -->

### Training the catalogue

Banking77, catalogue trained with `kv.train` on the public training split:

<!-- table:training -->
| Banking77, 300 test messages | Untrained catalogue | Trained catalogue |
|---|---|---|
| Catalogue alone (no LLM call) | 65.7% · 15 ms | 93.0% · 15 ms |
| Catalogue top-5, then LLM | 62.3% · 84 ms | 60.7% · 84 ms |
| Catalogue top-50, then LLM | 55.3% · 126 ms | 52.7% · 126 ms |
| Validation accuracy during training | 71.2% | 92.4% |
| Cost of training | – | 10,003 examples, 97 s in total (20 s of training, the rest is embedding the examples) |
<!-- /table:training -->

After training the catalogue is more accurate alone than followed by the LLM, and it answers in the time of one
embedding. The same catalogue configuration measured by two programs (`baselines.py` and `train_bench.py`) gives
56.3% and 55.3% for "top-50, then LLM": 3 messages out of 300.

A catalogue trained with the examples of half of the intents only:

<!-- table:heldout -->
| Intent of the test message | Untrained catalogue | Catalogue trained on half of the intents |
|---|---|---|
| Had training examples (39 intents) | 66.0% | 95.4% |
| Had no training example (38 intents) | 65.3% | 21.1% |
<!-- /table:heldout -->

### Reading the numbers

- Without training, kvindex is ahead of the OpenJev architecture on three tasks out of four and behind it on Typed
  Decisions (few options per question, five different questions per case).
- It keeps working where OpenJev stops (255 options) and where the list no longer fits in memory.
- CLM-8B with heads fine-tuned on the task is ahead of untrained kvindex where a training set exists (Banking77,
  Typed Decisions). The trained catalogue of kvindex was measured on Banking77 only.
- With 600 queries one accuracy has a standard error of about 2 points, with 150 queries about 4. Times vary
  between sessions on the same machine: the same configuration at 10,000 options measured between 84 and 99 ms on
  one day, with identical accuracy in every repetition.

## Limits

- Pure-attention models only (Qwen3, Llama, Mistral, Gemma). Models with recurrent layers cannot reuse a cached
  prefix.
- The cached list must fit in `n_ctx` and in GPU memory: about 78 KB per token for Qwen3-8B with `q8_0`.
- One list is cached at a time: alternating between lists recomputes the list that comes back.
- Accuracy on open vocabularies falls as the list grows (see the tables); the tool does not hide it.
- A trained head helps the options it has seen and hurts the others.

## Credits and prior work

kvindex combines techniques published by others. None of the mechanisms below is new here.

| Technique | Source |
|---|---|
| Candidates first in the prompt, encoded once thanks to prefix caching | Corpus-in-Context prompting, LOFT (Lee et al., 2024) |
| Reusing a precomputed KV cache across queries | Cache-Augmented Generation (Chan et al., 2024); TurboRAG, KVLink, CacheBlend |
| Decoding constrained to candidate names with a token trie | GENRE (De Cao et al., 2021) |
| Embedding shortlist followed by an LLM choice | the standard retrieve-then-rerank pipeline |
| Contrastive training of state and option projections | InfoNCE (van den Oord et al., 2018), CLIP (Radford et al., 2021), Contrastive Language Models (CLM-v0.1-8B) |
| Low-rank correction with an identity start | LoRA (Hu et al., 2021) |
| `Choice` / `Score` / `Noul` question types | OpenJev (MIT licence, © 2026 GPT-AGI). kvindex contains no OpenJev code: the types are written from scratch with the same names for compatibility. |

What this repository adds is the packaging: one interface that selects between the cached list and the catalogue
by size, exclusion of refuted options, an optional catalogue training that needs only numpy, and a measured
comparison that reports accuracy and time together.

Software, models and data used, each under its own licence and none redistributed here:
[llama.cpp](https://github.com/ggml-org/llama.cpp) and [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
(MIT), [Ollama](https://github.com/ollama/ollama) (MIT), NumPy (BSD), Qwen3 and Qwen3-Embedding (Apache-2.0),
[OpenJev](https://github.com/GPT-AGI/OpenJev) (MIT), the
[CLM-v0.1-8B](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B) heads,
[Banking77](https://github.com/PolyAI-LDN/task-specific-datasets) (CC BY 4.0), WordNet (Princeton WordNet licence)
and the Typed Decisions dataset (Apache-2.0).

## Repository layout

```
kvindex/        the library: core.py (cached index, typed questions), catalog.py (embedding catalogue),
                train.py (optional training), llamacpp.py and ollama.py (engines)
tests/          offline tests with a fake engine
examples/       quickstart.py
benchmarks/     baselines.py (comparison), scale.py (growing lists), train_bench.py (training),
                checks.py (correctness checks to run first), report.py (tables)
results/        raw results of every benchmark
paper/          article and figures
```

## License

MIT © 2026 Hobey Garrone
