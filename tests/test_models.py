"""Every architecture: loss is finite, gradients flow, generation returns valid images."""

import pytest
import torch

from tinyprism.registry import MODELS, TOKENIZERS, build_model, build_tokenizer
from tinyprism.text.tokenizer import WordTokenizer

TINY_UNET = dict(ch=16, mults=(1, 2), attn_levels=(1,), num_res=1, heads=2, text_dim=32, text_depth=1)
TINY = {
    "unet": TINY_UNET,
    "unet-eps": dict(TINY_UNET, prediction="eps"),
    "mar": dict(dim=32, depth=1, heads=2, head_width=32, head_depth=1, text_dim=32, text_depth=1),
}
TINY["transfusion"] = dict(dim=32, depth=1, heads=2)
TINY_DEFAULT = dict(dim=32, depth=1, heads=2, text_dim=32, text_depth=1)
TINY_VQ = dict(ch=16, z_dim=8, codebook_size=16)
FAST_SAMPLING = {
    "unet": dict(steps=3),
    "unet-eps": dict(steps=3),
    "maskgit": dict(steps=3),
    "mar": dict(num_iter=3, head_steps=2),
    "dit": dict(steps=2),
    "jit": dict(steps=2),
    "mmdit": dict(steps=2),
    "single-stream": dict(steps=2),
    "sana": dict(steps=2),
    "transfusion": dict(steps=2),
    "meanflow": dict(steps=1),
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


def test_meanflow_jvp_matches_finite_difference():
    """d/dt u(z_t, r, t) along (v, 0, 1) must equal a central finite difference."""
    from torch.func import jvp
    from torch.nn.attention import SDPBackend, sdpa_kernel

    torch.manual_seed(0)
    model = tiny_model("meanflow").double().eval()
    text = tuple(x.double() for x in model.encode_text(torch.zeros(2, 10, dtype=torch.long)))
    z, v = torch.randn(2, 3, 32, 32, dtype=torch.double), torch.randn(2, 3, 32, 32, dtype=torch.double)
    r, t = torch.tensor([0.2, 0.5], dtype=torch.double), torch.tensor([0.7, 0.9], dtype=torch.double)

    def fn(z_, r_, t_):
        return model.average_velocity(z_, r_, t_, text)

    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        _, dudt = jvp(fn, (z, r, t), (v, torch.zeros_like(r), torch.ones_like(t)))
        eps = 1e-5
        fd = (fn(z + eps * v, r, t + eps) - fn(z - eps * v, r, t - eps)) / (2 * eps)
    assert torch.allclose(dudt, fd, atol=1e-6, rtol=1e-4)


def test_transfusion_text_is_causal_and_blind_to_image():
    model = tiny_model("transfusion").eval()
    tokens = WordTokenizer().encode(["a large red circle at the top left"])
    text = model.encode_text(tokens)
    t = torch.full((1,), 0.5)
    h1, _, _ = model.backbone(torch.randn(1, 3, 32, 32), t, text)
    h2, _, _ = model.backbone(torch.randn(1, 3, 32, 32), t, text)
    assert torch.allclose(h1, h2, atol=1e-6)  # caption states never see the image
    tokens2 = tokens.clone()
    tokens2[0, -1] = 5
    h3, _, _ = model.backbone(torch.randn(1, 3, 32, 32), t, model.encode_text(tokens2))
    assert torch.allclose(h1[:, :-1], h3[:, :-1], atol=1e-6)  # changing the last token never affects earlier ones


def test_sana_linear_attention_matches_explicit_formula():
    from tinyprism.models.sana import LinearAttention

    torch.manual_seed(0)
    attn = LinearAttention(8, heads=1)
    x = torch.randn(1, 5, 8)
    q, k, v = attn.qkv(x).chunk(3, dim=-1)
    q, k = torch.relu(q), torch.relu(k)
    sim = q @ k.transpose(-2, -1)  # un-normalised kernel similarities
    ref = attn.proj((sim / (sim.sum(-1, keepdim=True) + 1e-6)) @ v)
    assert torch.allclose(attn(x), ref, atol=1e-5)


def test_mar_mask_ratio_is_truncated_not_clamped():
    from tinyprism.models.mar import MAR

    torch.manual_seed(0)
    r = MAR.sample_mask_ratio(20000, "cpu")
    assert r.min() >= 0.7 and r.max() <= 1.0
    assert (r == 0.7).float().mean() < 0.001  # clamping would put ~23% of the mass exactly at 0.7


def test_train_rejects_wrong_tokenizer(tmp_path):
    from tinyprism.registry import save_checkpoint
    from tinyprism.train import main

    vq = build_tokenizer("vq", **TINY_VQ)
    save_checkpoint(tmp_path / "vq.pt", "tokenizer", "vq", TINY_VQ, vq)
    with pytest.raises(SystemExit):
        main(["var", "--out", str(tmp_path / "var"), "--tokenizer", str(tmp_path / "vq.pt"), "--steps", "1"])


def test_transfusion_lm_targets_end_with_one_eos():
    from tinyprism.models.transfusion import Transfusion

    tokens = torch.tensor([[2, 5, 6, 0, 0], [0, 0, 0, 0, 0]])
    assert Transfusion.lm_targets(tokens).tolist() == [[5, 6, 0, -100], [-100, -100, -100, -100]]


@pytest.mark.parametrize("patchifier", ["unet", "linear"])
def test_transfusion_patchifiers(patchifier, batch):
    images, tokens = batch
    model = build_model("transfusion", dim=32, depth=1, heads=2, patchifier=patchifier)
    loss = model.loss(images, tokens)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.eval().generate(tokens[:1], steps=2).shape == (1, 3, 32, 32)
