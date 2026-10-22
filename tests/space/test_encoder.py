import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from amber.space.encoder import ClipEncoder, Encoder, pick_device


def solid(tmp_path: Path, colour: str, size: int = 224) -> Path:
    """Write a solid-colour PNG (`colour` is any PIL colour name) and return its path."""
    path = tmp_path / f"{colour}.png"
    Image.new("RGB", (size, size), colour).save(path)
    return path


# --- pick_device ---------------------------------------------------------------------


def test_pick_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert pick_device() == "cpu"


def test_pick_device_prefers_mps_then_cuda(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert pick_device() == "mps"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert pick_device() == "cuda"


# --- the Encoder protocol ------------------------------------------------------------


def test_fake_encoder_satisfies_the_encoder_protocol(fake_encoder):
    assert isinstance(fake_encoder, Encoder)
    assert not isinstance(object(), Encoder)


# --- ClipEncoder against a stub open_clip (no download, no real model) -----------------

STUB_DIM = 6
STUB_CONTEXT = 4


class _StubModel(torch.nn.Module):
    """Image 'features' are the per-channel pixel means padded to STUB_DIM; text features are
    the first STUB_CONTEXT token ids padded likewise. Deliberately not unit norm."""

    def __init__(self):
        super().__init__()
        self.calls: list[tuple[str, int]] = []

    def encode_image(self, batch):
        self.calls.append(("image", len(batch)))
        assert not torch.is_grad_enabled() and torch.is_inference_mode_enabled()
        means = batch.mean(dim=(2, 3))  # (b, 3)
        return torch.cat([means * 7.0, means.flip(1) + 1.0], dim=1)

    def encode_text(self, tokens):
        self.calls.append(("text", len(tokens)))
        assert not torch.is_grad_enabled() and torch.is_inference_mode_enabled()
        feats = tokens.float()[:, :STUB_CONTEXT]
        return torch.nn.functional.pad(feats, (0, STUB_DIM - STUB_CONTEXT)) + 1.0


@pytest.fixture
def stub_open_clip(monkeypatch):
    # ClipEncoder imports open_clip lazily, so a stub in sys.modules is what it will see
    seen: dict = {}
    model = _StubModel()

    def preprocess_train(_img):  # the training preprocess must never be used
        raise AssertionError("encoders must use the eval preprocess")

    def preprocess_val(img: Image.Image) -> torch.Tensor:
        assert img.mode == "RGB"
        arr = np.asarray(img.resize((8, 8)), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def tokenizer(texts):
        seen["tokenized"] = list(texts)
        out = torch.zeros((len(texts), STUB_CONTEXT), dtype=torch.long)
        for i, t in enumerate(texts):
            out[i, 0] = len(t)
            out[i, 1] = sum(map(ord, t)) % 11
        return out

    def create_model_and_transforms(name, pretrained=None, **kwargs):
        seen["create"] = (name, pretrained, kwargs)
        return model, preprocess_train, preprocess_val

    def get_tokenizer(name, **kwargs):
        seen["tokenizer"] = name
        return tokenizer

    fake = types.SimpleNamespace(
        create_model_and_transforms=create_model_and_transforms,
        get_tokenizer=get_tokenizer,
    )
    monkeypatch.setitem(sys.modules, "open_clip", fake)
    return types.SimpleNamespace(model=model, seen=seen)


def test_clip_encoder_wires_up_open_clip(stub_open_clip):
    enc = ClipEncoder("Stub-B", "stub_weights", device="cpu")
    name, pretrained, _ = stub_open_clip.seen["create"]
    assert (name, pretrained) == ("Stub-B", "stub_weights")
    assert stub_open_clip.seen["tokenizer"] == "Stub-B"
    assert (enc.model_name, enc.pretrained, enc.device) == (
        "Stub-B",
        "stub_weights",
        "cpu",
    )
    assert enc.dim == STUB_DIM and not stub_open_clip.model.training
    assert isinstance(enc, Encoder)


def test_clip_encoder_defaults_to_the_picked_device(stub_open_clip, monkeypatch):
    monkeypatch.setattr("amber.space.encoder.pick_device", lambda: "cpu")
    assert ClipEncoder("Stub-B", "stub_weights").device == "cpu"


def test_embed_images_batches_normalises_and_keeps_order(stub_open_clip, tmp_path):
    enc = ClipEncoder("Stub-B", "stub_weights", device="cpu", batch_size=2)
    paths = [
        solid(tmp_path, c, size=20) for c in ("red", "green", "blue", "white", "black")
    ]
    stub_open_clip.model.calls.clear()
    out = enc.embed_images(paths)
    assert out.shape == (5, STUB_DIM) and out.dtype == np.float32
    assert np.allclose(np.linalg.norm(out, axis=1), 1, atol=1e-5)
    assert [n for kind, n in stub_open_clip.model.calls if kind == "image"] == [2, 2, 1]
    one_by_one = np.vstack([enc.embed_images([p]) for p in paths])
    assert np.allclose(out, one_by_one, atol=1e-6)  # batching does not change results
    assert not np.allclose(out[0], out[1])  # and rows are in the order asked for


def test_embed_images_converts_to_rgb(stub_open_clip, tmp_path):
    path = tmp_path / "gray.png"
    Image.new("L", (16, 16), 128).save(
        path
    )  # the stub preprocess asserts mode == "RGB"
    assert ClipEncoder("Stub-B", "stub_weights", device="cpu").embed_images(
        [path]
    ).shape == (1, STUB_DIM)


def test_embed_texts_batches_and_normalises(stub_open_clip):
    enc = ClipEncoder("Stub-B", "stub_weights", device="cpu", batch_size=2)
    stub_open_clip.model.calls.clear()
    out = enc.embed_texts(["a", "bb", "ccc"])
    assert out.shape == (3, STUB_DIM) and out.dtype == np.float32
    assert np.allclose(np.linalg.norm(out, axis=1), 1, atol=1e-5)
    assert [n for kind, n in stub_open_clip.model.calls if kind == "text"] == [2, 1]
    assert stub_open_clip.seen["tokenized"] == ["ccc"]  # last batch


def test_empty_inputs_give_empty_arrays(stub_open_clip):
    enc = ClipEncoder("Stub-B", "stub_weights", device="cpu")
    for out in (enc.embed_images([]), enc.embed_texts([])):
        assert out.shape == (0, STUB_DIM) and out.dtype == np.float32


def test_batch_size_must_be_positive(stub_open_clip):
    with pytest.raises(ValueError):
        ClipEncoder("Stub-B", "stub_weights", device="cpu", batch_size=0)


# --- real model (downloads ViT-B-32 once, ~600 MB, into ~/.cache) ----------------------


@pytest.fixture(scope="module")
def clip():
    return ClipEncoder("ViT-B-32", "laion2b_s34b_b79k")


@pytest.mark.slow
def test_clip_encoder_color_sanity(clip, tmp_path):
    red, blue = clip.embed_images([solid(tmp_path, "red"), solid(tmp_path, "blue")])
    t = clip.embed_texts(["a red shirt"])[0]
    assert red.shape == (clip.dim,) and red @ t > blue @ t


@pytest.mark.slow
def test_clip_encoder_outputs_are_float32_unit_rows(clip, tmp_path):
    imgs = clip.embed_images([solid(tmp_path, c) for c in ("red", "green", "blue")])
    texts = clip.embed_texts(["a red shirt", "a blue shirt"])
    assert imgs.shape == (3, clip.dim) and texts.shape == (2, clip.dim)
    assert imgs.dtype == texts.dtype == np.float32
    assert np.allclose(np.linalg.norm(np.vstack([imgs, texts]), axis=1), 1, atol=1e-4)
    assert (clip.model_name, clip.pretrained, clip.dim) == (
        "ViT-B-32",
        "laion2b_s34b_b79k",
        512,
    )


@pytest.mark.slow
def test_clip_encoder_batch_size_does_not_change_results(clip, tmp_path):
    paths = [solid(tmp_path, c) for c in ("red", "green", "blue")]
    small = ClipEncoder("ViT-B-32", "laion2b_s34b_b79k", batch_size=1)
    assert np.allclose(small.embed_images(paths), clip.embed_images(paths), atol=1e-4)
