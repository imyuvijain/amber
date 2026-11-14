"""Cluster topic documents into named topics (see the topics v1 spec).

Spherical k-means on the unique texts (weighted by how often each occurs), then
near-duplicate centroids merge, each cluster is named from a generic label vocabulary
(or by its c-TF-IDF keywords when no label fits), and clusters sharing a label merge.
Documents far from every centroid, and topics that end up tiny, stay unassigned (-1).
"""

import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from amber.topics.labels import LABELS

SEED = 7
K_MIN, K_MAX = 20, 80
MERGE_COS = 0.75
ASSIGN_COS = 0.3
NAME_COS = 0.3
MIN_DOCS = 30
N_KEYWORDS = 8
_TOKEN = re.compile(r"[^\W\d_]{3,}", re.UNICODE)
STOPWORDS = frozenset(
    [
        "the",
        "and",
        "for",
        "with",
        "you",
        "your",
        "this",
        "that",
        "from",
        "how",
        "what",
        "why",
        "who",
        "are",
        "was",
        "were",
        "have",
        "has",
        "not",
        "but",
        "all",
        "can",
        "will",
        "just",
        "our",
        "out",
        "new",
        "get",
        "got",
        "one",
        "two",
        "its",
        "it's",
        "into",
        "about",
        "more",
        "best",
        "top",
        "full",
        "part",
        "video",
        "official",
        "episode",
        "ep",
        "vs",
        "feat",
        "ft",
        "live",
        "day",
        "days",
        "ki",
        "ka",
        "ke",
        "hai",
        "aur",
        "se",
        "to",
        "of",
        "in",
        "on",
        "is",
        "a",
        "an",
        "my",
        "me",
        "we",
        "us",
        "they",
        "them",
        "he",
        "she",
        "his",
        "her",
        "be",
        "do",
        "does",
        "did",
        "http",
        "https",
        "www",
        "com",
    ]
)


@dataclass
class TopicModel:
    doc_topic: np.ndarray  # per document: index into ``topics`` or -1
    topics: pd.DataFrame  # topic_id, name, named_by, utility, keywords, n_docs
    centroids: np.ndarray  # (n_topics, dim), unit rows


def choose_k(n_unique: int) -> int:
    return int(min(K_MAX, max(K_MIN, round(math.sqrt(n_unique / 10)))))


def _unit(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(n == 0, 1, n)


def _centroids(X: np.ndarray, w: np.ndarray, lab: np.ndarray, n: int) -> np.ndarray:
    C = np.zeros((n, X.shape[1]))
    np.add.at(C, lab, X * w[:, None])
    return _unit(C)


def _relabel(lab: np.ndarray, groups: list[int]) -> tuple[np.ndarray, int]:
    """Map old cluster -> group id (dense 0..g-1)."""
    uniq = {g: i for i, g in enumerate(dict.fromkeys(groups))}
    return np.array([uniq[groups[c]] for c in lab]), len(uniq)


def _union(n: int, pairs) -> list[int]:
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return [find(i) for i in range(n)]


def _keywords(
    texts: list[str], counts: np.ndarray, lab: np.ndarray, n: int
) -> list[list[str]]:
    """c-TF-IDF: a word scores high if it's frequent in this topic and rare elsewhere."""
    tf = [Counter() for _ in range(n)]
    for t, c, g in zip(texts, counts, lab):
        for w in _TOKEN.findall(t.lower()):
            if w not in STOPWORDS:
                tf[g][w] += c
    total = Counter()
    for c in tf:
        total.update(c)
    avg = np.mean([sum(c.values()) for c in tf]) or 1.0
    out = []
    for c in tf:
        size = sum(c.values()) or 1
        scored = sorted(
            ((v / size * math.log(1 + avg / total[w]), w) for w, v in c.items()),
            key=lambda s: (-s[0], s[1]),
        )
        out.append([w for _, w in scored[:N_KEYWORDS]])
    return out


def extract_topics(
    docs: pd.DataFrame, encoder, labels=LABELS, k: int | None = None, min_docs: int = MIN_DOCS
) -> TopicModel:
    """Cluster ``docs["text"]`` into topics. ``encoder.embed_texts`` returns unit rows."""
    codes, uniq = pd.factorize(docs["text"], sort=True)
    texts = [str(t) for t in uniq]
    counts = np.bincount(codes, minlength=len(texts)).astype(float)
    X = _unit(np.asarray(encoder.embed_texts(texts), dtype=np.float64))
    k = min(k or choose_k(len(texts)), len(texts))
    km = KMeans(n_clusters=k, random_state=SEED, n_init=3).fit(X, sample_weight=counts)
    lab = km.labels_
    C = _centroids(X, counts, lab, k)

    # 1. merge near-duplicate centroids
    sim = C @ C.T
    groups = _union(
        k, [(a, b) for a in range(k) for b in range(a + 1, k) if sim[a, b] >= MERGE_COS]
    )
    lab, n = _relabel(lab, groups)
    C = _centroids(X, counts, lab, n)

    # 2. name by label; clusters sharing a label merge
    names = [lbl for lbl, _ in labels]
    L = (
        _unit(np.asarray(encoder.embed_texts(names), dtype=np.float64))
        if names
        else np.zeros((0, X.shape[1]))
    )
    best = (C @ L.T) if len(names) else np.zeros((n, 0))
    pick = [
        int(best[i].argmax()) if len(names) and best[i].max() >= NAME_COS else None
        for i in range(n)
    ]
    groups = [("label", p) if p is not None else ("own", i) for i, p in enumerate(pick)]
    lab, n = _relabel(lab, groups)
    group_label = list(dict.fromkeys(groups))
    C = _centroids(X, counts, lab, n)

    # 3. assign documents; drop far ones and tiny topics
    cos = (X * C[lab]).sum(1)
    text_topic = np.where(cos >= ASSIGN_COS, lab, -1)
    doc_t = text_topic[codes]
    sizes = np.bincount(doc_t[doc_t >= 0], minlength=n)
    keep = [g for g in np.argsort(-sizes, kind="stable") if sizes[g] >= min_docs]
    remap = np.full(n + 1, -1)
    remap[keep] = np.arange(len(keep))
    doc_t = remap[doc_t]
    text_t = remap[text_topic]

    ok = text_t >= 0
    kw = _keywords(
        [t for t, o in zip(texts, ok) if o], counts[ok], text_t[ok], len(keep)
    )
    rows = []
    for new, g in enumerate(keep):
        kind, p = group_label[g]
        labelled = kind == "label"
        rows.append(
            {
                "topic_id": f"t{new:02d}",
                "name": labels[p][0] if labelled else " · ".join(kw[new][:3]),
                "named_by": "label" if labelled else "keywords",
                "utility": bool(labels[p][1]) if labelled else False,
                "keywords": kw[new],
                "n_docs": int(sizes[g]),
            }
        )
    topics = pd.DataFrame(
        rows, columns=["topic_id", "name", "named_by", "utility", "keywords", "n_docs"]
    )
    return TopicModel(doc_t, topics, C[keep] if keep else np.zeros((0, X.shape[1])))
