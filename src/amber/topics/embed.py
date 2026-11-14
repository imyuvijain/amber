"""Text encoders for topics: a small local sentence model, a model-free hashing
fallback, and a disk cache so each text is embedded once.

``MiniLMEncoder`` runs ``sentence-transformers/all-MiniLM-L6-v2`` (about 90 MB, fetched
once into the Hugging Face cache). Nothing leaves the machine.
"""

import hashlib
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np

MINILM = "sentence-transformers/all-MiniLM-L6-v2"
_TOKEN = re.compile(r"\w+", re.UNICODE)


def _key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class HashingEncoder:
    """Bag of words hashed into ``dim`` buckets (no model). For tests and fallback."""

    model_name = "hashing"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in _TOKEN.findall(t.lower()):
                h = int(hashlib.md5(w.encode("utf-8")).hexdigest()[:8], 16)
                out[i, h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        out[norms[:, 0] == 0, 0] = 1.0
        return out / np.linalg.norm(out, axis=1, keepdims=True)


class MiniLMEncoder:
    model_name = MINILM

    def __init__(self, device: str | None = None, batch_size: int = 256):
        from sentence_transformers import SentenceTransformer

        from amber.space.encoder import pick_device

        self.model = SentenceTransformer(MINILM, device=device or pick_device())
        self.dim = self.model.get_sentence_embedding_dimension()
        self.batch_size = batch_size

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        return self.model.encode(
            list(texts), batch_size=self.batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        ).astype(np.float32)


class CachedEncoder:
    """Wraps an encoder; vectors are kept by text hash in an ``.npz`` (float16)."""

    def __init__(self, encoder, path: Path):
        self.encoder, self.path = encoder, Path(path)
        self.model_name = getattr(encoder, "model_name", "?")
        self.cache: dict[str, np.ndarray] = {}
        if self.path.exists():
            z = np.load(self.path, allow_pickle=False)
            if str(z["model"]) == self.model_name:
                self.cache = dict(zip(z["keys"].tolist(), z["vecs"].astype(np.float32)))

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        keys = [_key(t) for t in texts]
        todo = list(dict.fromkeys(t for t, k in zip(texts, keys) if k not in self.cache))
        if todo:
            for t, v in zip(todo, self.encoder.embed_texts(todo)):
                self.cache[_key(t)] = np.asarray(v, dtype=np.float16).astype(np.float32)
        return np.stack([self.cache[k] for k in keys]) if keys else np.zeros((0, 1), np.float32)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(self.cache)
        np.savez(
            self.path, model=np.array(self.model_name), keys=np.array(keys),
            vecs=np.stack([self.cache[k] for k in keys]).astype(np.float16),
        )
