"""Encoders map photos and text into one shared, L2-normalised vector space.

`Encoder` is the structural interface the rest of Amber codes against; `ClipEncoder` is the
real implementation on top of open_clip, and the test suite swaps in a model-free fake with
the same shape. `model_name` and `pretrained` identify the embedding space an encoder
produces (vectors from different models must never be mixed).
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@runtime_checkable
class Encoder(Protocol):
    """Both `embed_*` methods return `(n, dim)` float32 arrays with unit-norm rows."""

    dim: int
    model_name: str
    pretrained: str

    def embed_images(self, paths: Sequence[Path]) -> np.ndarray: ...

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray: ...


def pick_device() -> str:
    """The fastest torch device available here: "mps", then "cuda", then "cpu"."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _unit_rows(features: torch.Tensor) -> np.ndarray:
    """L2-normalise model outputs and hand them back as float32 numpy on the CPU."""
    return F.normalize(features.float(), dim=-1).cpu().numpy()


class ClipEncoder:
    """A CLIP-style model from open_clip, e.g. `ClipEncoder("ViT-B-32", "laion2b_s34b_b79k")`.

    Weights are fetched into open_clip's cache (under `~/.cache`) on first use. Inputs are
    processed in batches of `batch_size`; `device` defaults to `pick_device()`.
    """

    def __init__(
        self,
        model_name: str,
        pretrained: str,
        device: str | None = None,
        batch_size: int = 64,
    ):
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        # imported here: open_clip (with torchvision and timm) takes seconds to import,
        # and most users of this module only need `pick_device` or the `Encoder` protocol
        import open_clip

        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device or pick_device()
        self.batch_size = batch_size
        model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self._model = model.eval()
        self._tokenizer = open_clip.get_tokenizer(model_name)
        self.dim: int = self._encode_texts([""]).shape[1]  # read off a real output

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        tokens = self._tokenizer(texts).to(self.device)
        with torch.inference_mode():
            return _unit_rows(self._model.encode_text(tokens))

    def _encode_images(self, paths: list[Path]) -> np.ndarray:
        tensors = []
        for path in paths:
            with Image.open(path) as image:
                tensors.append(self._preprocess(image.convert("RGB")))
        batch = torch.stack(tensors).to(self.device)
        with torch.inference_mode():
            return _unit_rows(self._model.encode_image(batch))

    def embed_images(self, paths: Sequence[Path]) -> np.ndarray:
        """Unit-norm embeddings `(n, dim)`, float32, of the images at `paths`, in order."""
        paths = list(paths)
        parts = [
            self._encode_images(paths[i : i + self.batch_size])
            for i in range(0, len(paths), self.batch_size)
        ]
        return np.concatenate(parts) if parts else self._empty()

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Unit-norm embeddings `(n, dim)`, float32, of `texts`, in order."""
        texts = list(texts)
        parts = [
            self._encode_texts(texts[i : i + self.batch_size])
            for i in range(0, len(texts), self.batch_size)
        ]
        return np.concatenate(parts) if parts else self._empty()

    def _empty(self) -> np.ndarray:
        return np.zeros((0, self.dim), dtype=np.float32)
