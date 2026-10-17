import torch

from tinyprism.data.shapes import NUM_CLASSES, ShapesDataset, caption, render
from tinyprism.text.encoder import TextEncoder
from tinyprism.text.tokenizer import WordTokenizer


def test_render_shapes_and_range():
    img = render(0, 0, 4, 1, res=32)
    assert img.shape == (32, 32, 3) and img.dtype.name == "uint8"
    # Centre pixel of a large red circle at the center is red.
    assert img[16, 16, 0] > 200 and img[16, 16, 2] < 80


def test_all_shapes_render_distinct():
    imgs = [render(0, s, 4, 1) for s in range(NUM_CLASSES[1])]
    for i in range(len(imgs)):
        for j in range(i + 1, len(imgs)):
            assert (imgs[i] != imgs[j]).any()


def test_dataset_batch():
    b = ShapesDataset(seed=0).sample(8)
    assert b.images.shape == (8, 3, 32, 32)
    assert b.images.min() >= -1 and b.images.max() <= 1
    assert b.labels.shape == (8, 4)
    assert b.captions[0] == caption(*b.labels[0].tolist())


def test_tokenizer_roundtrip_and_null():
    tok = WordTokenizer()
    text = "a large red circle at the top left"
    ids = tok.encode([text, ""])
    assert ids.shape == (2, tok.max_len)
    assert tok.decode(ids)[0] == text
    assert (ids[1] == 0).all()  # empty prompt == null prompt
    assert 1 not in ids[0].tolist()  # no <unk> for in-grammar captions


def test_text_encoder_shapes():
    tok = WordTokenizer()
    enc = TextEncoder(tok.vocab_size, tok.max_len, dim=32, depth=1)
    seq, pooled = enc(tok.encode(["a small blue square at the left"]))
    assert seq.shape == (1, tok.max_len, 32) and pooled.shape == (1, 32)
    assert torch.isfinite(seq).all()


def test_bank_matches_labels():
    data = ShapesDataset(seed=0, variants=2)
    b = data.sample(16)
    fresh = data.render_labels(b.labels.numpy()).images
    # Jitter differs, but the dominant colour of every image must agree with a fresh render.
    assert torch.allclose(b.images.mean((2, 3)), fresh.mean((2, 3)), atol=0.08)
