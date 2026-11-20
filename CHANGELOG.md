# Changelog

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
