"""Offline tests: a fake Ollama client stands in for the server."""
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from decidekv import KV, Choice, Noul, Score  # noqa: E402
from decidekv.catalog import Catalog, TopKIndex  # noqa: E402
from decidekv.core import Index, parse_options  # noqa: E402


class FakeClient:
    """Backend-contract fake: picks the option whose text appears in the state; logs every prompt."""

    def __init__(self):
        self.host, self.prompts = "http://fake", []
        self.firsts = []
        self.calls = []

    def architecture(self):
        return "llama"

    @staticmethod
    def _parse(segments):
        first, second, _ = segments
        if first.startswith("You are a decision engine. Questions about"):
            prefix, state_seg = second, first.split("\n\n", 1)[1]
        else:
            prefix, state_seg = first, second
        state = state_seg.split("\n", 1)[1].lower()                  # "### <state label>\n<state>"
        body = prefix.split("\n\n### ", 1)[1]                         # "<Nouns>\n<option lines>"
        opts = body.split("\n", 1)[1].split("\n\n", 1)[0].strip().splitlines()
        return state, opts

    def prefill(self, segments, num_ctx=None):
        self.prompts.append("".join(segments))
        return {"input_tokens": 50, "cached_tokens": 0, "output_tokens": 1}

    def label_logprobs(self, segments, labels, num_ctx=None):
        self.firsts.append(segments[0])
        self.prompts.append("".join(segments))
        state, opts = self._parse(segments)
        texts = [o.split(": ")[-1].lower() for o in opts]          # description part of "L: key: desc"
        hit = next((i for i, t in enumerate(texts) if t in state or state in t), len(labels) - 1)
        lp = {lab: math.log(0.7 if i == hit else 0.3 / (len(labels) - 1)) for i, lab in enumerate(labels)}
        return lp, {"input_tokens": 10, "cached_tokens": 40, "output_tokens": 1}

    def pick_option(self, segments, options, num_ctx=None, enum_max=None, allowed=None):
        self.firsts.append(segments[0])
        self.prompts.append("".join(segments))
        state, lines = self._parse(segments)
        desc = {ln[2:].split(": ", 1)[0]: ln[2:].split(": ", 1)[-1].lower() for ln in lines}
        pool = list(allowed) if allowed is not None else list(options)
        best = next((o for o in pool if o.lower() in state or desc.get(o, "") in state), pool[-1])
        return best, math.log(0.8), {"input_tokens": 10, "cached_tokens": 90, "output_tokens": 3}

    def score_options(self, segments, options, allowed=None):
        self.firsts.append(segments[0])
        self.prompts.append("".join(segments))
        state, lines = self._parse(segments)
        desc = {ln[2:].split(": ", 1)[0]: ln[2:].split(": ", 1)[-1].lower() for ln in lines}
        pool = list(allowed) if allowed is not None else list(options)
        best = next((o for o in pool if o.lower() in state or desc.get(o, "") in state), pool[-1])
        if len(pool) == 1:
            return ({pool[0]: math.log(1.0)}, {"input_tokens": 10, "cached_tokens": 90, "output_tokens": 1})
        return ({o: math.log(0.7) if o == best else math.log(0.3 / (len(pool) - 1)) for o in pool},
                {"input_tokens": 10, "cached_tokens": 90, "output_tokens": len(pool)})

    def hold(self, prefix):
        self.calls.append(("hold", prefix))

    def release(self, prefix):
        self.calls.append(("release", prefix))


WORDS = ["zebra", "milk", "piano", "hammer", "window", "garden", "rocket", "butter"]


class FakeEncoder:
    """Letter-count vectors of the text after 'Query:' (or of the whole text), L2-normalised."""

    def __call__(self, texts):
        out = []
        for t in texts:
            v = np.zeros(26, dtype=np.float32)
            for ch in t.split("Query:")[-1].lower():
                if "a" <= ch <= "z":
                    v[ord(ch) - 97] += 1
            out.append(v / (np.linalg.norm(v) + 1e-12))
        return np.stack(out)



def make_kv(**kw):
    kv = KV("fake", backend="ollama", **kw)
    kv.client = FakeClient()
    return kv


def test_choose_labels_and_probabilities():
    kv = make_kv()
    idx = kv.index(["card arrival", "lost card", "refund"], labels=True)
    a = idx.choose("where is my card arrival status")
    assert a.choice == "card arrival"
    assert abs(sum(a.probabilities.values()) - 1) < 1e-6
    assert a.probabilities["card arrival"] > 0.5


def test_prefix_is_identical_across_states():
    kv = make_kv()
    idx = kv.index(["alpha", "beta", "gamma"])
    idx.choose("state one")
    idx.choose("a completely different state")
    p1, p2 = kv.client.prompts[-2], kv.client.prompts[-1]
    cut = p1.index("### State")
    assert p1[:cut] == p2[:cut] == idx.prefix + idx.question_block(None)


def test_index_is_reused():
    kv = make_kv()
    assert kv.index(["a", "b"]) is kv.index(["a", "b"])


def test_name_mode_for_many_options():
    kv = make_kv()
    opts = [f"item {i}" for i in range(300)] + ["golden retriever"]
    idx = kv.index(opts)
    assert idx.labels is None
    assert idx.choose("my dog is a golden retriever").choice == "golden retriever"


def test_decide_typed_questions():
    kv = make_kv()
    r = kv.decide("customer is furious, card lost", {
        "dept": Choice(instructions="Which team?", options={"cards": "card lost", "loans": "mortgage"}),
        "mood": Score(instructions="Mood?", criteria=["calm", "annoyed", "furious"]),
        "urgent": Noul(instructions="Urgent?"),
    })
    assert r.answers["dept"].choice == "cards"
    assert r.answers["mood"].score > 1.0
    assert 0.0 <= r.answers["urgent"].noul <= 1.0
    assert r.usage.cached_tokens > 0


def test_decide_accepts_plain_dicts():
    kv = make_kv()
    r = kv.decide({"ticket": "refund please"}, {"q": {"type": "choice", "instructions": "Which?",
                                                      "options": ["refund please", "other"]}})
    assert r.answers["q"].choice == "refund please"


def test_domain_prompt_is_cached_with_names():
    kv = make_kv()
    q = "Which intent does this customer message express?"
    idx = kv.index(["card arrival", "lost card"], noun="intent", state_label="Customer message", question=q)
    seg = idx.segments("my card is lost", q)
    assert "### Intents" in idx.prefix and "intent" in idx.prefix
    assert seg[0] == idx.prefix + f"### Question\n{q}\n\n"
    assert seg[1].startswith("### Customer message") and seg[2] == "### Intent\n"
    old = kv.index(["card arrival", "lost card"], noun="intent", state_label="Customer message", question=q,
                   question_first=False).segments("my card is lost", q)
    assert old[0] == idx.prefix and old[2] == f"### Question\n{q}\n\n### Intent\n"
    sf = idx.segments("my card is lost", q, order="state-first")      # same blocks, other order
    assert sorted("".join(sf)) == sorted("".join(seg)) and "".join(sf) != "".join(seg)
    assert idx.choose("my lost card").choice == "lost card"


def test_exclude_makes_refuted_options_impossible():
    kv = make_kv()
    idx = kv.index(["red door", "blue door", "green door"])
    assert idx.choose("open the red door").choice == "red door"
    a = idx.choose("open the red door", exclude=["red door"])      # refuted: must pick something else
    assert a.choice != "red door"
    lab = kv.index(["red door", "blue door", "green door"], labels=True)
    assert lab.choose("open the red door", exclude=["red door"]).choice != "red door"


def test_parse_options():
    keys, descs, texts = parse_options(["a", "b"])
    assert keys == ["a", "b"]
    assert descs == ["", ""]
    assert texts == ["a", "b"]

    keys, descs, texts = parse_options({"a": "first", "b": ""})
    assert keys == ["a", "b"]
    assert descs == ["first", ""]
    assert texts == ["a: first", "b"]

    with pytest.raises(ValueError):
        parse_options(["a"])
    with pytest.raises(ValueError):
        parse_options(["a", "a"])


def test_selector_by_size():
    kv = make_kv(embed=FakeEncoder(), flat_max=3)
    idx = kv.index(WORDS[:3])
    assert isinstance(idx, Index)

    idx = kv.index(WORDS)
    assert isinstance(idx, TopKIndex)
    assert len(idx) == 8


def test_many_options_need_an_embedding_model():
    kv = make_kv(flat_max=3)
    with pytest.raises(ValueError):
        kv.index(WORDS)


def test_topk_shows_only_the_closest_options():
    kv = make_kv(embed=FakeEncoder(), flat_max=3)
    idx = kv.index(WORDS, k=3)
    a = idx.choose("zebra")
    assert a.choice == "zebra"
    assert len(idx.last_candidates) == 3
    assert idx.last_candidates[0] == "zebra"
    assert "- zebra" in kv.client.prompts[-1]
    assert kv.client.prompts[-1].count("- ") == 3


def test_topk_exclude():
    kv = make_kv(embed=FakeEncoder(), flat_max=3)
    idx = kv.index(WORDS, k=3)
    a = idx.choose("zebra", exclude=["zebra"])
    assert a.choice != "zebra"
    assert "zebra" not in idx.last_candidates
    assert len(idx.last_candidates) == 3

    with pytest.raises(ValueError):
        idx.choose("zebra", exclude=WORDS)


def test_catalog_head_is_applied():
    class Swap:
        def options(self, m):
            return m[:, ::-1].copy()

        def state(self, m):
            return m[:, ::-1].copy()

    texts = ["ab", "yz"]
    c = Catalog(texts, FakeEncoder())
    assert c.top(c.query("ab", "q"), 1) == [0]

    c2 = Catalog(texts, FakeEncoder(), head=Swap())
    assert c2.top(c2.query("ab", "q"), 1) == [0]
    assert np.allclose(c2.vec, c.vec[:, ::-1])


def letter_examples():
    """Four options and 160 examples whose states share no letter with their option: untrained similarity is zero."""
    options = ["aaaa", "bbbb", "cccc", "dddd"]
    examples = [("wxyz"[i % 4] * (3 + i % 4) + "mnop"[(i // 4) % 4], options[i % 4]) for i in range(160)]
    return options, examples


def test_train_improves_the_catalogue():
    options, examples = letter_examples()
    kv = make_kv(embed=FakeEncoder())
    head = kv.train(options, examples)
    assert head.report["val_start"] < 0.5 and head.report["val_best"] == 1.0
    idx = kv.index(options, head=head, k=1)
    assert isinstance(idx, TopKIndex)
    assert [idx.choose(s).choice for s in ("wwww", "xxxxx", "yyy", "zzzzzz")] == options
    assert kv.client.prompts == []


def test_trained_head_is_saved_and_loaded(tmp_path):
    options, examples = letter_examples()
    kv = make_kv(embed=FakeEncoder())
    path = str(tmp_path / "head.npz")
    kv.train(options, examples, out=path)
    idx = kv.index(options, head=path, k=1)
    assert idx.choose("yyyy").choice == "cccc"


def test_train_rejects_examples_outside_the_option_list():
    kv = make_kv(embed=FakeEncoder())
    with pytest.raises(ValueError):
        kv.train(["aaaa", "bbbb"], [("www", "zzzz")])


def test_query_template_is_configurable():
    class RecordingEncoder(FakeEncoder):
        def __init__(self):
            self.seen = []

        def __call__(self, texts):
            self.seen += list(texts)
            return super().__call__(texts)

    enc = RecordingEncoder()
    kv = make_kv(embed=enc, flat_max=3, query_template="search: {state}")
    kv.index(WORDS, k=3).choose("zebra")
    assert enc.seen[-1] == "search: zebra"

    enc = RecordingEncoder()
    kv = make_kv(embed=enc, flat_max=3)
    kv.index(WORDS, k=3, question="Q?").choose("zebra")
    assert enc.seen[-1] == "Instruct: Q?\nQuery:zebra"


def test_catalog_vectors_are_cached_on_disk(tmp_path):
    enc = FakeEncoder()
    texts = ["apple", "banana", "cherry"]
    v1 = Catalog.cached(texts, enc, str(tmp_path))
    assert len(list(tmp_path.glob("*.npy"))) == 1
    v2 = Catalog.cached(texts, enc, str(tmp_path))
    assert np.array_equal(v1, v2)
    assert len(list(tmp_path.glob("*.npy"))) == 1
    v3 = Catalog.cached(texts + ["date"], enc, str(tmp_path))
    assert v3.shape[0] == 4
    assert len(list(tmp_path.glob("*.npy"))) == 2


def test_full_probabilities_cover_every_option():
    kv = make_kv()
    idx = kv.index(["card arrival", "lost card", "refund"])
    a = idx.choose("where is my card arrival status", full=True)
    assert a.choice == "card arrival"
    assert set(a.probabilities) == {"card arrival", "lost card", "refund"}
    assert abs(sum(a.probabilities.values()) - 1) < 1e-6
    assert a.probabilities["card arrival"] > 0.6
    b = idx.choose("where is my card arrival status")
    assert b.choice == "card arrival" and set(b.probabilities) == {"card arrival"}


def test_scores_respect_exclusions():
    kv = make_kv()
    idx = kv.index(["card arrival", "lost card", "refund"])
    s = idx.scores("where is my card arrival status", allowed=["lost card", "refund"])
    assert set(s) == {"lost card", "refund"}
    assert abs(sum(math.exp(v) for v in s.values()) - 1) < 1e-6
    a = idx.choose("where is my card arrival status", exclude=["card arrival"], full=True)
    assert a.choice != "card arrival" and a.probabilities["card arrival"] == 0.0


def typed_examples():
    """States made of the letters w, x, y, z; the right answers use other letters, so the untrained catalogue
    cannot find them by similarity."""
    choice = Choice(instructions="Which item?", options=["aaaa", "bbbb", "cccc", "dddd"])
    score = Score(instructions="How much?", criteria=["eeee", "ffff", "gggg"])
    noul = Noul(instructions="Is it so?", criteria={"true": "hhhh", "false": "iiii"})
    examples = []
    for i in range(240):
        s = "wxyz"[i % 4] * (3 + i % 3) + "mnop"[(i // 4) % 4]
        examples.append((s, choice, ["aaaa", "bbbb", "cccc", "dddd"][i % 4]))
        examples.append((s, score, {"0": 0.8, "1": 0.2} if i % 4 < 2 else {"2": 0.9, "1": 0.1}))
        examples.append((s, noul, i % 4 in (0, 3)))
    return choice, score, noul, examples


def test_train_questions_learns_choice_score_and_noul():
    choice, score, noul, examples = typed_examples()
    kv = make_kv(embed=FakeEncoder())
    head = kv.train_questions(examples, epochs=60, batch=32, lr=0.01)
    assert head.report["val_best"] > head.report["val_start"]
    r = kv.decide("wwww", {"c": choice, "s": score, "n": noul}, head=head)
    assert r.answers["c"].choice == "aaaa"
    assert r.answers["s"].score < 1.0
    assert r.answers["n"].noul > 0.5
    r2 = kv.decide("yyyy", {"c": choice, "s": score, "n": noul}, head=head)
    assert r2.answers["c"].choice == "cccc"
    assert r2.answers["s"].score > 1.0
    assert r2.answers["n"].noul < 0.5
    assert kv.client.prompts == []


def test_head_file_and_soft_answers(tmp_path):
    choice, score, noul, examples = typed_examples()
    kv = make_kv(embed=FakeEncoder())
    path = str(tmp_path / "typed.npz")
    kv.train_questions(examples, out=path, epochs=60, batch=32, lr=0.01)
    r = kv.decide("xxxx", {"c": choice}, head=path)
    assert r.answers["c"].choice == "bbbb"
    assert abs(sum(r.answers["c"].probabilities.values()) - 1) < 1e-4


def test_answer_targets():
    from decidekv.core import answer_target
    noul = Noul(instructions="q")
    score = Score(instructions="q", criteria=["a", "b", "c"])
    assert answer_target(noul, ["Yes", "No"], True) == [1.0, 0.0]
    assert answer_target(noul, ["Yes", "No"], 0.25) == [0.25, 0.75]
    assert answer_target(noul, ["Yes", "No"], {"true": 3, "false": 1}) == [0.75, 0.25]
    assert answer_target(score, ["0", "1", "2"], 2) == [0.0, 0.0, 1.0]
    assert answer_target(score, ["0", "1", "2"], {0: 1, 2: 1}) == [0.5, 0.0, 0.5]
    with pytest.raises(ValueError):
        answer_target(score, ["0", "1", "2"], {"7": 1.0})


def shared_calls(kv):
    """hold/release calls of a state shared by several questions (options-first prefixes are held too)."""
    return [c for c in kv.client.calls if c[1].startswith("You are a decision engine. Questions about")]


def test_decide_state_order_shares_the_state():
    kv = make_kv()
    qs = {"n1": Noul(instructions="Is the light red?"), "n2": Noul(instructions="Is it raining?"),
          "s": Score(instructions="How loud is it?", criteria=["quiet", "loud"])}
    state = "A long street at night. " * 20
    r_opt = kv.decide(state, qs, order="options")
    assert shared_calls(kv) == []
    n = len(kv.client.firsts)
    r_st = kv.decide(state, qs, order="state")
    firsts = kv.client.firsts[n:]
    assert len(firsts) == 3
    assert len(set(firsts)) == 1
    assert firsts[0].startswith("You are a decision engine. Questions about")
    assert shared_calls(kv) == [("hold", firsts[0]), ("release", firsts[0])]
    for qid in qs:
        assert r_opt.answers[qid] == r_st.answers[qid]


def test_decide_auto_order():
    kv = make_kv()
    two = {"a": Noul(instructions="Is the light red?"), "b": Noul(instructions="Is it raining?")}
    kv.decide("A long street at night. " * 20, two, order="auto")
    assert [c[0] for c in shared_calls(kv)] == ["hold", "release"]
    kv.client.calls.clear()
    kv.decide("A long street at night. " * 20, {"a": two["a"]}, order="auto")
    assert shared_calls(kv) == []
    kv.decide("Night.", two, order="auto")
    assert [c[0] for c in shared_calls(kv)] == ["hold", "release"]
    kv.client.calls.clear()
    words = [f"word{i:04d}" for i in range(400)]
    big = {"a": Choice(instructions="Which word?", options=words), "b": Choice(instructions="Which other word?",
                                                                               options=words)}
    kv.decide("Night.", big, order="auto")
    assert shared_calls(kv) == []


def test_decide_default_order_is_options():
    kv = make_kv()
    two = {"a": Noul(instructions="Is the light red?"), "b": Noul(instructions="Is it raining?")}
    r = kv.decide("Night.", two)
    assert shared_calls(kv) == []
    assert set(r.question_ms) == {"a", "b"} and all(v >= 0 for v in r.question_ms.values())


def test_decide_unknown_order():
    kv = make_kv()
    with pytest.raises(ValueError):
        kv.decide("x", {"a": Noul(instructions="q")}, order="sideways")


def test_fallback_for_options_without_examples():
    kv = make_kv(embed=FakeEncoder())
    seen = WORDS[:4]
    examples = [(f"{w} {w} {w}", w) for w in seen for _ in range(40)]
    head = kv.train(WORDS, examples, epochs=60, batch=32, lr=0.01)
    assert head.report["trained_options"] == [0, 1, 2, 3]
    idx = kv.index(WORDS, head=head)
    for w in WORDS:
        assert idx.choose(f"{w} {w} {w}").choice == w
    assert idx.fallbacks == len(kv.client.prompts)
    off = kv.index(WORDS, head=head, fallback=0)
    n = len(kv.client.prompts)
    for w in WORDS:
        off.choose(f"{w} {w} {w}")
    assert len(kv.client.prompts) == n
    assert off.fallbacks == 0


def test_no_fallback_when_every_option_was_trained():
    kv = make_kv(embed=FakeEncoder())
    examples = [(f"{w} {w} {w}", w) for w in WORDS for _ in range(20)]
    head = kv.train(WORDS, examples, epochs=40, batch=32, lr=0.01)
    idx = kv.index(WORDS, head=head)
    for w in WORDS:
        idx.choose(f"{w} {w}")
    assert idx.fallbacks == 0
    assert kv.client.prompts == []


def fake_server(slots=3, reply=None):
    """A LlamaServer that never opens a connection: `_post` records the request and returns `reply`."""
    from decidekv.llamaserver import LlamaServer
    srv = LlamaServer.__new__(LlamaServer)
    srv.host, srv.timeout, srv.process, srv.slots, srv.n_ctx = "http://fake", 1, None, slots, 4096
    srv._wrap, srv.scratch, srv.model_path = ("<u>", "</u>"), slots - 1, ""
    from collections import OrderedDict
    srv._slot_of = OrderedDict()
    srv.sent = []

    def post(path, body):
        srv.sent.append((path, body))
        return reply or {}
    srv._post = post
    return srv


def test_server_gbnf_literal():
    from decidekv.llamaserver import _gbnf_literal
    assert _gbnf_literal('say "hi"\\now') == '"say \\"hi\\"\\\\now"'
    assert _gbnf_literal("card arrival") == '"card arrival"'


def test_server_slots_lru():
    srv = fake_server(slots=3)
    srv.prefill(["A", "x"])
    srv.prefill(["B", "x"])
    assert srv._slot_of == {"A": 0, "B": 1}
    srv.prefill(["A", "y"])
    srv.prefill(["C", "x"])
    assert dict(srv._slot_of) == {"A": 0, "C": 1}
    assert srv.sent[-1][1]["id_slot"] == 1
    srv.prefill(["D", "x"])
    assert dict(srv._slot_of) == {"C": 1, "D": 0}
    srv.release("C")
    srv.label_logprobs(["C", "q"], ["Yes", "No"])
    assert srv.sent[-1][1]["id_slot"] == 2
    assert srv.sent[-1][1]["prompt"] == "<u>Cq</u>"


def test_server_label_logprobs():
    reply = {"completion_probabilities": [{"token": "Yes", "logprob": -0.1, "top_logprobs": [
        {"token": "Yes", "logprob": math.log(0.6)}, {"token": " Yes", "logprob": math.log(0.1)},
        {"token": "No", "logprob": math.log(0.2)}]}], "timings": {"prompt_n": 5, "cache_n": 40}, "tokens_predicted": 1}
    srv = fake_server(reply=reply)
    lp, st = srv.label_logprobs(["s", "q"], ["Yes", "No", "Maybe"])
    assert abs(math.exp(lp["Yes"]) - 0.7) < 1e-9
    assert abs(math.exp(lp["No"]) - 0.2) < 1e-9
    assert lp["Maybe"] == -30.0
    assert st == {"input_tokens": 5, "cached_tokens": 40, "output_tokens": 1}
    assert srv.sent[0][1]["n_probs"] == 100
    assert srv.sent[0][1]["n_predict"] == 1


def test_server_pick_option():
    reply = {"content": "lost card", "completion_probabilities": [{"token": "lost", "logprob": -0.5},
            {"token": " card", "logprob": -0.25}, {"token": "", "logprob": -0.01}], "timings": {},
             "tokens_predicted": 3}
    srv = fake_server(reply=reply)
    options = ["card arrival", "lost card", "refund"]
    text, lp, st = srv.pick_option(["s", "q"], options, allowed=["lost card", "refund"])
    assert text == "lost card"
    assert abs(lp + 0.75) < 1e-12
    assert srv.sent[0][1]["grammar"] == 'root ::= "lost card" | "refund"'
    with pytest.raises(NotImplementedError):
        srv.score_options(["s", "q"], ["a", "b"])


def write_manifest(path, layers):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"layers": layers}), encoding="utf8")


def test_ollama_gguf_v1_and_v2_layouts(tmp_path, monkeypatch):
    from decidekv.llamacpp import ollama_gguf
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path))
    write_manifest(tmp_path / "manifests" / "registry.ollama.ai" / "library" / "qwen3" / "8b",
                  [{"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:aaa"}])
    write_manifest(tmp_path / "manifests-v2" / "ollama.com" / "library" / "mine" / "latest",
                  [{"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:bbb"}])
    write_manifest(tmp_path / "manifests-v2" / "hf.co" / "user" / "repo-gguf" / "iq2_s",
                  [{"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:ccc"}])
    assert ollama_gguf("qwen3:8b") == os.path.join(str(tmp_path), "blobs", "sha256-aaa")
    assert ollama_gguf("mine") == os.path.join(str(tmp_path), "blobs", "sha256-bbb")
    assert ollama_gguf("hf.co/user/repo-GGUF:IQ2_S") == os.path.join(str(tmp_path), "blobs", "sha256-ccc")
    with pytest.raises(FileNotFoundError):
        ollama_gguf("missing:1b")


def test_ollama_gguf_vision_model_skips_projector(tmp_path, monkeypatch):
    from decidekv.llamacpp import ollama_gguf
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path))
    write_manifest(tmp_path / "manifests-v2" / "ollama.com" / "library" / "vis" / "iq2_s",
                  [{"mediaType": "application/vnd.ollama.image.projector", "digest": "sha256:proj"},
                   {"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:model"}])
    assert ollama_gguf("vis:iq2_s") == os.path.join(str(tmp_path), "blobs", "sha256-model")


class TreeClient(FakeClient):
    """FakeClient with the batched method: answers every item with label_logprobs and records the batches."""

    def __init__(self):
        super().__init__()
        self.trees = []
        self.prefills = 0

    def prefill(self, segments, num_ctx=None):
        self.prefills += 1
        return super().prefill(segments, num_ctx)

    def label_logprobs_tree(self, items):
        self.trees.append(len(items))
        return [self.label_logprobs(segments, labels) for segments, labels in items]


def test_decide_batches_label_questions_that_share_the_state():
    qs = {"n1": Noul(instructions="Is the light red?"), "n2": Noul(instructions="Is it raining?"),
          "n3": Noul(instructions="Is it late?"),
          "s": Score(instructions="How loud is it?", criteria=["quiet", "loud"])}
    state = "A long street at night. " * 20
    plain = make_kv()
    r_plain = plain.decide(state, qs, order="state")
    kv = make_kv()
    kv.client = TreeClient()
    r_tree = kv.decide(state, qs, order="state")
    assert kv.client.trees == [4]
    for qid in qs:
        assert r_tree.answers[qid] == r_plain.answers[qid]
    assert kv.client.prefills == 0
    assert r_tree.usage.output_tokens == r_plain.usage.output_tokens


def test_decide_without_tree_method_or_single_question():
    kv = make_kv()
    kv.client = TreeClient()
    kv.decide("A long street at night. " * 20, {"a": Noul(instructions="Is it late?")}, order="state")
    assert kv.client.trees == []
    kv.decide("A long street at night. " * 20, {"a": Noul(instructions="Is it late?"),
                                                "b": Noul(instructions="Is it red?")}, order="options")
    assert kv.client.trees == []
    assert kv.client.prefills >= 1


def test_cascade_threshold():
    choice, score, noul, examples = typed_examples()
    kv = make_kv(embed=FakeEncoder())
    head = kv.train_questions(examples, epochs=60, batch=32, lr=0.01)
    qs = {"c": choice, "s": score, "n": noul}
    r0 = kv.decide("wwww", qs, head=head, threshold=0.0)
    assert r0.escalated == []
    assert kv.client.prompts == []
    r1 = kv.decide("wwww", qs, head=head, threshold=1.01)
    assert sorted(r1.escalated) == ["c", "n", "s"]
    assert len(kv.client.prompts) > 0
    plain = make_kv(embed=FakeEncoder()).decide("wwww", qs)
    for qid in qs:
        assert r1.answers[qid] == plain.answers[qid]
    assert r1.model.endswith("/cascade")


def test_train_questions_per_question_layout():
    choice, score, noul, examples = typed_examples()
    kv = make_kv(embed=FakeEncoder())
    head = kv.train_questions(examples, layout="per-question", epochs=60, batch=32, lr=0.01)
    assert head.report["layout"] == "per-question"
    r = kv.decide("yyyy", {"c": choice}, head=head)
    assert r.answers["c"].choice == "cccc"
    with pytest.raises(ValueError):
        kv.train_questions(examples, layout="sideways")


def test_semantic_cache_in_choose():
    kv = make_kv(embed=FakeEncoder())
    idx = kv.index(WORDS)
    first = idx.choose("zebra zebra", cache=0.99)
    n = len(kv.client.prompts)
    again = idx.choose("zebra zebra", cache=0.99)
    assert again == first
    assert len(kv.client.prompts) == n
    assert idx.semantic.hits == 1
    idx.choose("piano piano", cache=0.99)
    assert len(kv.client.prompts) > n


def test_semantic_cache_in_decide():
    kv = make_kv(embed=FakeEncoder())
    qs = {"a": Noul(instructions="Is the light red?"), "b": Noul(instructions="Is it raining?")}
    r1 = kv.decide("A long street at night. " * 20, qs, cache=0.99)
    assert r1.cached == []
    n = len(kv.client.prompts)
    r2 = kv.decide("A long street at night. " * 20, qs, cache=0.99)
    assert sorted(r2.cached) == ["a", "b"]
    assert len(kv.client.prompts) == n
    assert r2.answers == r1.answers


def test_cached_encoder_reuses_vectors():
    from decidekv.catalog import CachedEncoder
    calls = []

    def enc(texts):
        calls.append(list(texts))
        return np.array([[len(t), 1.0] for t in texts])
    c = CachedEncoder(enc, max_items=3)
    assert c(["a", "bb", "a"]).shape == (3, 2) and calls == [["a", "bb"]]
    c(["a", "ccc"])
    assert calls[-1] == ["ccc"]
    c(["dddd"])
    assert len(c.vectors) == 3


def test_thresholds_and_answer_matches():
    from decidekv.primitives import ChoiceAnswer, NoulAnswer, ScoreAnswer
    from decidekv.train import answer_matches, thresholds_for_target
    rec = [("q", 0.9, True)] * 18 + [("q", 0.2, False)] * 4 + [("r", 0.9, True)] * 5
    th = thresholds_for_target(rec, 0.95)
    assert abs(th["q"] - 0.9) < 1e-9 and th["r"] == 1.01
    assert answer_matches(ChoiceAnswer("x", {"x": 1.0}, 1.0), "x")
    assert answer_matches(ScoreAnswer(score=1.0, probabilities={"0": 0.2, "1": 0.8}, confidence=0.5, legend={}), 1)
    assert answer_matches(NoulAnswer(noul=0.7), True) and not answer_matches(NoulAnswer(noul=0.7), False)


def test_calibrated_cascade_and_escalate_order():
    choice, score, noul, examples = typed_examples()
    kv = make_kv(embed=FakeEncoder())
    head = kv.train_questions(examples, epochs=60, batch=32, lr=0.01)
    qs = {"c": choice, "s": score, "n": noul}
    with pytest.raises(ValueError):
        kv.decide("wwww", qs, head=head, threshold="calibrated")
    held = []
    for i in range(40):
        s = "wxyz"[i % 4] * (3 + i % 3) + "mnop"[(i // 4) % 4]
        held.append((s, qs, {"c": ["aaaa", "bbbb", "cccc", "dddd"][i % 4], "s": "0" if i % 4 < 2 else "2",
                             "n": i % 4 in (0, 3)}))
    th = head.calibrate(kv, held, target=0.9, min_n=5)
    assert set(th) == {"c", "s", "n"} and "calibration" in head.report
    r = kv.decide("wwww", qs, head=head, threshold="calibrated")
    assert set(r.answers) == {"c", "s", "n"}
    kv.client.calls.clear()
    r = kv.decide("wwww", qs, head=head, threshold=1.01, escalate_order="state")
    assert sorted(r.escalated) == ["c", "n", "s"] and [c[0] for c in shared_calls(kv)] == ["hold", "release"]
    kv.client.calls.clear()
    kv.decide("wwww", qs, head=head, threshold=1.01)
    assert shared_calls(kv) == []


def test_warmup():
    choice, score, noul, examples = typed_examples()
    kv = make_kv(embed=FakeEncoder())
    head = kv.train_questions(examples, epochs=10, batch=32, lr=0.01)
    ms = kv.warmup({"c": choice, "s": score, "n": noul}, head=head)
    assert isinstance(ms, float) and ms >= 0


class ResidentClient(TreeClient):
    """TreeClient with the options-first batched method: answers every item with label_logprobs and records
    the batches."""

    def __init__(self):
        super().__init__()
        self.resident_calls = []

    def label_logprobs_resident(self, items):
        self.resident_calls.append(len(items))
        return [self.label_logprobs(segments, labels) for segments, labels in items]


def test_decide_batches_label_questions_options_first():
    qs = {"n1": Noul(instructions="Is the light red?"), "n2": Noul(instructions="Is it raining?"),
          "s": Score(instructions="How loud is it?", criteria=["quiet", "loud"])}
    state = "A long street at night. " * 20
    r_plain = make_kv().decide(state, qs, order="options")
    kv = make_kv()
    kv.client = ResidentClient()
    r = kv.decide(state, qs, order="options")
    assert kv.client.resident_calls == [3]
    assert kv.client.trees == []
    for qid in qs:
        assert r.answers[qid] == r_plain.answers[qid]
    assert r.usage.output_tokens == r_plain.usage.output_tokens
    assert set(r.question_ms) == set(qs)


def test_batch_options_switch_and_single_question():
    kv = make_kv()
    kv.client = ResidentClient()
    kv.batch_options = False
    kv.decide("A street.", {"a": Noul(instructions="Is it late?"), "b": Noul(instructions="Is it red?")},
              order="options")
    assert kv.client.resident_calls == []
    kv.batch_options = True
    kv.decide("A street.", {"a": Noul(instructions="Is it late?")}, order="options")
    assert kv.client.resident_calls == []
    kv.decide("A street.", {"a": Noul(instructions="Is it late?"), "b": Noul(instructions="Is it red?")},
              order="state")
    assert kv.client.resident_calls == []
    assert kv.client.trees == [2]


def test_lists_order_prompt_and_batch():
    kv = make_kv()
    qs = [Score(instructions="How loud?", criteria=["quiet", "medium", "loud"]), Noul(instructions="Is it late?"),
          Noul(instructions="Is it red?")]
    entries = [tuple(kv.question_index(q, warm=False))[1:] for q in qs]
    reqs = kv._lists_requests("A street.", entries)
    assert len(reqs) == 3 and len({segs[:2] for segs, _ in reqs}) == 1
    first = reqs[0][0][0]
    assert first.count("### List A") == 1 and first.count("### List B") == 1 and "### List C" not in first
    assert "list A" in reqs[0][0][2] and "list B" in reqs[1][0][2] and "Is it late?" in reqs[1][0][2]
    assert set(reqs[1][1]) == {"Yes", "No"}
    kv.client = TreeClient()
    qd = {"s": qs[0], "a": qs[1], "b": qs[2], "c": Choice(instructions="Which?", options=["x", "y"])}
    r = kv.decide("A long street at night, late.", qd, order="lists")
    assert kv.client.trees == [3]                   # the Choice by name is asked options first
    assert set(r.answers) == set(qd) and set(r.question_ms) == set(qd)
    kv.decide("A street.", {"a": qs[1]}, order="lists")
    assert kv.client.trees == [3]                   # one question alone: no batch
