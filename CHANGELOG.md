# Changelog

## 1.4.0 (2026-11-25)

Changes suggested by use in an application on llama-server (Gemma 4 12B).

- **llama-server: label questions on several slots at once** (`LlamaServer.label_logprobs_tree`, also used for
  options first). The number of slots keeps the work that the extra slots repeat (the state, recomputed in each slot)
  within `parallel_budget` characters (2,048). Qwen3-8B, 8 slots, a turn of 15 questions about a short state: 407 →
  286 ms options first, 381 → 333 ms state first, 372 → 323 ms `lists`; a long state stays on one slot, unchanged
  (sending every question in parallel regardless made it twice as slow). Typed Decisions: 51.4% against 51.7%
  (−0.3 ± 0.9). `kv.client.parallel = False` restores 1.3.
- **llama-server: labels on by default** when the options fit one-token labels (`labels=None`), so that every option
  gets a probability; names on the other engines, as before. A warning above 52 options.
- **Chat wrapper**: asked to llama-server (`/apply-template`), so templates with their own tags work;
  `KV(..., wrap=(before, after))` sets it by hand (llama.cpp and llama-server), and an unrenderable template now gives
  a clear error pointing to it.
- **`answer_prefix`**: text added at the start of the answer, for models that always reason.
- **Threads**: calls from several threads on one `KV` (and its indexes) take turns under a lock.
- **`order="lists"` takes only `Score` and `Noul` questions**; `Choice` questions are asked options first. With
  letter labels (now the default on llama-server) their labels were confused with the list names: Typed Decisions on
  llama-server 42.8% (Choice 26%); with the fix 50.6% (−1.3 ± 1.7 against options first).

## 1.3.0 (2026-11-20)

- **`order="lists"`** in `KV.decide`: the distinct option lists of the call go first (List A, List B, ...) in one
  prefix that stays cached while the lists do not change, then the state once, then every question with the name of
  its list, all in parallel in one prefix-tree pass (questions answered by labels; the others are asked options
  first). It keeps the accuracy of the options-first order at the speed of the state-first order. Typed Decisions
  (2,000 decisions), against options first: Qwen3-8B 50.1% vs 51.2% (paired difference −1.1 ± 1.5 points), 198 vs
  319 ms per case; Qwen3.8-27B 68.7% vs 69.1% (−0.4 ± 1.9), 975 vs 1,753 ms. A turn of 15 questions about a long state:
  326 vs 2,760 ms on Qwen3-8B, 1,527 vs 11,314 ms on the 27B (prefix tree: 339 and 1,628 ms).
- Two other variants were measured and left out: every question with its options first and only "answer to question
  k" after the state (the fastest, but −6.0 ± 2.2 points on Qwen3-8B), and the same with the question repeated after
  the state (−3.0 ± 2.2). See the paper, negative results.
- `Index.option_lines()`: the option lines of an index without the heading (used by `options_block`).

## 1.2.0 (2026-11-20)

- **Options first, in parallel**: in `KV.decide` with `order="options"`, the questions answered by labels (`Score`,
  `Noul`, `Choice` with `labels=True`) run together in parallel sequences of one llama.cpp pass, each copied from its
  own resident option list (new backend method `label_logprobs_resident`). One turn of 15 questions about a short
  state: Qwen3-8B 338 → 185 ms, Qwen3.8-27B 1,266 → 933 ms; with a long state no change (every question still computes
  the state after its own list). Typed Decisions accuracy unchanged (51.1% both, 2,000 decisions; individual
  probabilities can differ slightly, as between any two batch layouts in llama.cpp). `kv.batch_options = False`
  restores the one-at-a-time behaviour.

## 1.1.0 (2026-11-19)

Changes suggested by use in an application with a 27B model.

- **Default prompt order is now `order="options"`** (was `"auto"`): every option list stays first and resident; it
  was the more accurate order in most tests. `order="state"` (state first, one prefix-tree pass) remains the fast
  choice for long states, and `order="auto"` is still available.
- **Cascade**: `escalate_order` chooses the order in which the uncertain questions are asked to the LLM (default
  `"options"`); `threshold="calibrated"` uses one threshold per question found by the new `Head.calibrate(kv,
  held_out, target=0.95)` (lowest confidence with at least `target` agreement on held-out examples; saved with the
  head).
- **`KV.warmup(questions, head=None)`**: prepares indexes, resident prefixes, encoder and option vectors at start-up,
  so the first call is not slower than the others.
- **`Response.question_ms`**: time of every question (an estimate for the questions answered together in one pass).
- **Encoder cache**: the embedding of a text already seen is reused (also for an encoder passed as a function).
- The package version is now consistent (`decidekv` 1.1.0); an environment that still lists `kvindex 0.1.0` should
  reinstall with `pip uninstall kvindex` and `pip install -e <path to the repository>`.
- Documented that, with the default decoding by name, a `Choice` answer carries only the probability of the chosen
  option (`labels=True` or `full=True` give every option).

## 1.0.1 (2026-11-17)

First public version (Zenodo DOI 10.5281/zenodo.24323634).
