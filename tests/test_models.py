"""Every architecture: loss is finite, gradients flow, generation returns valid images."""

import pytest
import torch

from tinyprism.registry import MODELS, TOKENIZERS, build_model, build_tokenizer
from tinyprism.text.tokenizer import WordTokenizer

TINY = {
    "unet": dict(ch=16, mults=(1, 2), attn_levels=(1,), num_res=1, heads=2, text_dim=32, text_depth=1),
    "mar": dict(dim=32, depth=1, heads=2, head_width=32, head_depth=1, text_dim=32, text_depth=1),
}
TINY_DEFAULT = dict(dim=32, depth=1, heads=2, text_dim=32, text_depth=1)
TINY_VQ = dict(ch=16, z_dim=8, codebook_size=16)
FAST_SAMPLING = {
    "unet": dict(steps=3),
    "maskgit": dict(steps=3),
    "mar": dict(num_iter=3, head_steps=2),
    "dit": dict(steps=2),
    "jit": dict(steps=2),
    "mmdit": dict(steps=2),
    "single-stream": dict(steps=2),
}


def tiny_model(name):
    spec = MODELS[name]
    vq_config = None
    if spec.recipe.tokenizer:
        vq_config = {**TOKENIZERS[spec.recipe.tokenizer].kwargs, **TINY_VQ}
    return build_model(name, vq_config=vq_config, **TINY.get(name, TINY_DEFAULT))


@pytest.fixture(scope="module")
def batch():
    from tinyprism.data.shapes import ShapesDataset

    b = ShapesDataset(seed=0).sample(4)
    return b.images, WordTokenizer().encode(b.captions)


@pytest.mark.parametrize("name", list(MODELS))
def test_loss_and_backward(name, batch):
    torch.manual_seed(0)
    images, tokens = batch
    model = tiny_model(name).train()
    loss = model.loss(images, tokens)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads), "every trainable parameter should receive a gradient"
    assert all(p.grad is None for p in getattr(model, "vq", torch.nn.Linear(1, 1)).parameters() if not p.requires_grad)


@pytest.mark.parametrize("name", list(MODELS))
def test_generate(name, batch):
    torch.manual_seed(0)
    _, tokens = batch
    model = tiny_model(name).eval()
    images = model.generate(tokens[:2], cfg_scale=2.0, generator=torch.Generator().manual_seed(0), **FAST_SAMPLING.get(name, {}))
    assert images.shape == (2, 3, 32, 32)
    assert torch.isfinite(images).all() and images.min() >= -1 and images.max() <= 1


@pytest.mark.parametrize("scales", [(8,), (1, 2, 4, 8)])
def test_vq_tokenizer(scales):
    vq = build_tokenizer("vq", scales=scales, **TINY_VQ)
    x = torch.rand(2, 3, 32, 32) * 2 - 1
    recon, vq_loss, idx, _ = vq(x)
    assert recon.shape == x.shape and torch.isfinite(vq_loss)
    assert [i.shape[1] for i in idx] == [s * s for s in scales]
    # decode(encode(x)) must equal the forward reconstruction (eval mode, deterministic).
    vq.eval()
    with torch.no_grad():
        assert torch.allclose(vq.decode(vq.encode(x)), vq(x)[0].clamp(-1, 1), atol=1e-5)
        if len(scales) > 1:
            assert vq.var_inputs(vq.encode(x)).shape == (2, sum(s * s for s in scales[1:]), 8)


def test_var_mask_is_block_causal():
    model = tiny_model("var")
    lvl = model.lvl
    mask = model.attn_mask
    assert mask[0].sum() == 1  # the start token only sees itself
    assert mask[-1].all()  # the last scale sees everything
    assert not mask[lvl == 0][:, lvl == 1].any()  # coarse never sees fine


def test_dit_overfits_single_prompt():
    """Sanity check that the training loop actually learns: loss drops on a fixed batch."""
    torch.manual_seed(0)
    from tinyprism.data.shapes import ShapesDataset

    b = ShapesDataset(seed=0, jitter=False).sample(8)
    tokens = WordTokenizer().encode(b.captions)
    model = build_model("dit", dim=64, depth=2, heads=2, text_dim=32, text_depth=1)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    losses = []
    for _ in range(60):
        loss = model.loss(b.images, tokens)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert sum(losses[-10:]) / 10 < 0.75 * sum(losses[:10]) / 10
