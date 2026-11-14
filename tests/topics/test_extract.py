import hashlib

import numpy as np
import pandas as pd

from amber.topics.extract import choose_k, extract_topics

GROUPS = {
    "cooking": ["recipe", "bake", "bread", "curry", "cooking"],
    "physics": ["quantum", "relativity", "physics", "electron", "gravity"],
    "anime": ["anime", "manga", "naruto", "episode", "otaku"],
}
FILLER = ["the", "best", "how", "to", "my", "new", "video", "part", "2", "full"]


class KeywordEncoder:
    """Group words light up the group's dim; anything else is small seeded noise."""

    dim = 16

    def embed_texts(self, texts):
        out = []
        for t in texts:
            v = np.random.default_rng(int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)).standard_normal(self.dim) * 0.05
            for g, (name, words) in enumerate(GROUPS.items()):
                v[g] += sum(w in t.lower().split() for w in words)
            out.append(v / np.linalg.norm(v))
        return np.array(out, dtype=np.float32)


def _docs(n_per=120, seed=7):
    rng = np.random.default_rng(seed)
    rows, truth = [], []
    for g, words in enumerate(GROUPS.values()):
        for _ in range(n_per):
            toks = list(rng.choice(words, 2)) + list(rng.choice(FILLER, 3))
            rng.shuffle(toks)
            rows.append(" ".join(toks))
            truth.append(g)
    return pd.DataFrame({"text": rows, "weight": 1.0}), np.array(truth)


def test_choose_k_is_clamped():
    assert choose_k(10) == 20
    assert choose_k(10**7) == 80
    assert 20 <= choose_k(40_000) <= 80


def test_extract_recovers_planted_groups_and_names_them_by_label():
    docs, truth = _docs()
    model = extract_topics(docs, KeywordEncoder(), labels=[("cooking", False), ("physics", False), ("anime", False), ("weather", True)], k=6)
    topic = model.doc_topic
    assigned = topic >= 0
    assert assigned.mean() > 0.9
    # purity: each found topic is (almost) one planted group
    for t in np.unique(topic[assigned]):
        share = np.bincount(truth[topic == t], minlength=3).max() / (topic == t).sum()
        assert share > 0.9
    names = set(model.topics["name"])
    assert {"cooking", "physics", "anime"} <= names
    assert len(model.topics) == 3  # same-label clusters merged
    assert not model.topics["utility"].any()


def test_extract_falls_back_to_keywords_when_no_label_fits():
    docs, _ = _docs()
    model = extract_topics(docs, KeywordEncoder(), labels=[("weather", True)], k=3)
    assert (model.topics["named_by"] == "keywords").all()
    assert all(" · " in n for n in model.topics["name"])


def test_extract_is_deterministic():
    docs, _ = _docs()
    a = extract_topics(docs, KeywordEncoder(), labels=[("cooking", False)], k=5)
    b = extract_topics(docs, KeywordEncoder(), labels=[("cooking", False)], k=5)
    assert (a.doc_topic == b.doc_topic).all()
