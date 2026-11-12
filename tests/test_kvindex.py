"""Offline tests: a fake Ollama client stands in for the server."""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from kvindex import KV, Choice, Noul, Score  # noqa: E402
from kvindex.catalog import Catalog, TopKIndex  # noqa: E402
from kvindex.core import Index, parse_options  # noqa: E402


class FakeClient:
    """Backend-contract fake: picks the option whose text appears in the state; logs every prompt."""

    def __init__(self):
        self.host, self.prompts = "http://fake", []

    def architecture(self):
        return "llama"

    @staticmethod
    def _parse(segments):
        prefix, state_seg, _ = segments
        state = state_seg.split("\n", 1)[1].lower()                  # "### <state label>\n<state>"
        body = prefix.split("\n\n### ", 1)[1]                         # "<Nouns>\n<option lines>"
        opts = body.split("\n", 1)[1].strip().splitlines()
        return state, opts

    def prefill(self, segments, num_ctx=None):
        self.prompts.append("".join(segments))
        return {"input_tokens": 50, "cached_tokens": 0, "output_tokens": 1}

    def label_logprobs(self, segments, labels, num_ctx=None):
        self.prompts.append("".join(segments))
        state, opts = self._parse(segments)
        texts = [o.split(": ")[-1].lower() for o in opts]          # description part of "L: key: desc"
        hit = next((i for i, t in enumerate(texts) if t in state or state in t), len(labels) - 1)
        lp = {lab: math.log(0.7 if i == hit else 0.3 / (len(labels) - 1)) for i, lab in enumerate(labels)}
        return lp, {"input_tokens": 10, "cached_tokens": 40, "output_tokens": 1}

    def pick_option(self, segments, options, num_ctx=None, enum_max=None, allowed=None):
        self.prompts.append("".join(segments))
        state, lines = self._parse(segments)
        desc = {ln[2:].split(": ", 1)[0]: ln[2:].split(": ", 1)[-1].lower() for ln in lines}
        pool = list(allowed) if allowed is not None else list(options)
        best = next((o for o in pool if o.lower() in state or desc.get(o, "") in state), pool[-1])
        return best, math.log(0.8), {"input_tokens": 10, "cached_tokens": 90, "output_tokens": 3}


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
    assert p1[:cut] == p2[:cut] == idx.prefix


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
    assert seg[1].startswith("### Customer message") and seg[2] == f"### Question\n{q}\n\n### Intent\n"
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
