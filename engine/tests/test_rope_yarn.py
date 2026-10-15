"""``--rope-yarn`` (YaRN for Flash Next on CUDA) without a GPU: the tables against the reference formula, plain RoPE's
tables unchanged, and the window the capacity plan, the CLI and the server see."""

import json
import math
from types import SimpleNamespace

import pytest

from tensorfold import cli, rope_scaling
from tensorfold.cuda.capacity import Geometry, Weights as CapWeights, choose, make_plan
from tests.test_cuda_capacity import fake_runtime  # noqa: F401  (the fixture)

torch = pytest.importorskip("torch")

# Flash Next's text config, as far as the rope reads it
TEXT = {"max_position_embeddings": 262144, "head_dim": 256,
        "rope_parameters": {"rope_type": "default", "type": "default", "rope_theta": 10_000_000,
                            "partial_rotary_factor": 0.25, "mrope_interleaved": True, "mrope_section": [11, 11, 10]}}


def reference_yarn(base: float, dim: int, factor: float, original: int, beta_fast=None, beta_slow=None,
                   truncate=True, attention_factor=None, mscale=None, mscale_all_dim=None):
    """transformers 5.18 ``modeling_rope_utils._compute_yarn_parameters``, transcribed (float32, as it runs)."""

    def get_mscale(scale, mscale=1):
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    if attention_factor is None:
        if mscale and mscale_all_dim:
            attention_factor = float(get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim))
        else:
            attention_factor = get_mscale(factor)
    beta_fast = beta_fast or 32
    beta_slow = beta_slow or 1

    def find_correction_dim(num_rotations, dim, base, max_position_embeddings):
        return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_position_embeddings, truncate):
        low = find_correction_dim(low_rot, dim, base, max_position_embeddings)
        high = find_correction_dim(high_rot, dim, base, max_position_embeddings)
        if truncate:
            low = math.floor(low)
            high = math.ceil(high)
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        return torch.clamp(linear_func, 0, 1)

    pos_freqs = base ** (torch.arange(0, dim, 2).to(dtype=torch.float) / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)
    low, high = find_correction_range(beta_fast, beta_slow, dim, base, original, truncate)
    inv_freq_extrapolation_factor = 1 - linear_ramp_factor(low, high, dim // 2).to(dtype=torch.float)
    inv_freq = (inv_freq_interpolation * (1 - inv_freq_extrapolation_factor)
                + inv_freq_extrapolation * inv_freq_extrapolation_factor)
    return inv_freq, attention_factor


def _cfg(yarn):
    return SimpleNamespace(rope_theta=1e7, rotary_dim=64, yarn=yarn)


@pytest.mark.parametrize("factor", [2.0, 4.0, 1.5])
def test_yarn_tables_equal_the_reference(factor):
    yarn = rope_scaling.yarn_parameters(TEXT, factor)
    inv, scale = rope_scaling.rope_tables(_cfg(yarn))
    ref_inv, ref_scale = reference_yarn(1e7, 64, factor, 262144)
    assert inv.dtype == torch.float32 and inv.shape == (32,)
    # the engine rounds float64 math to fp32; the reference runs in fp32: one or two fp32 ulps apart at most
    assert torch.allclose(inv, ref_inv, rtol=3e-7, atol=0)
    assert scale == pytest.approx(ref_scale, rel=1e-15)
    assert scale == pytest.approx(0.1 * math.log(factor) + 1.0)


def test_yarn_at_factor_2_keeps_fast_pairs_and_halves_slow_ones():
    plain, one = rope_scaling.rope_tables(_cfg(None))
    yarn = rope_scaling.yarn_parameters(TEXT, 2.0)
    inv, scale = rope_scaling.rope_tables(_cfg(yarn))
    low, high = rope_scaling.correction_range(yarn, 64, 1e7)
    assert (low, high) == (14, 22)          # beta_fast 32, beta_slow 1 over the 262,144-token pretraining window
    assert one == 1.0 and scale == pytest.approx(1.0693147180559945)
    assert torch.allclose(inv[:low], plain[:low], rtol=2e-7)          # high frequencies: extrapolated (unchanged)
    assert torch.allclose(inv[high:], plain[high:] / 2, rtol=2e-7)    # low frequencies: interpolated by the factor
    assert bool((inv[low + 1:high] < plain[low + 1:high]).all())    # the ramp blends the two between


def test_checkpoint_yarn_parameters_are_honoured():
    text = dict(TEXT, rope_parameters={**TEXT["rope_parameters"], "rope_type": "yarn", "type": "yarn", "factor": 4.0,
                                       "original_max_position_embeddings": 262144, "beta_fast": 16,
                                       "beta_slow": 2, "truncate": False})
    yarn = rope_scaling.yarn_parameters(text)
    assert yarn["factor"] == 4.0 and rope_scaling.window(text) == 1048576
    inv = rope_scaling.yarn_inv_freq(1e7, 64, yarn).float()
    ref, scale = reference_yarn(1e7, 64, 4.0, 262144, beta_fast=16, beta_slow=2, truncate=False)
    assert torch.allclose(inv, ref, rtol=3e-7, atol=0) and rope_scaling.attention_factor(yarn) == pytest.approx(scale)
    # --rope-yarn replaces the factor, keeping the checkpoint's other parameters
    over = rope_scaling.yarn_parameters(text, 2.0)
    assert over["factor"] == 2.0 and over["beta_fast"] == 16 and rope_scaling.window(text, 2.0) == 524288
    # an explicit attention factor, and the mscale pair, as the reference reads them
    assert rope_scaling.attention_factor({**yarn, "attention_factor": 1.25}) == 1.25
    _, ref_m = reference_yarn(1e7, 64, 4.0, 262144, mscale=1.0, mscale_all_dim=0.5)
    assert rope_scaling.attention_factor({**yarn, "mscale": 1.0, "mscale_all_dim": 0.5}) == pytest.approx(ref_m)


def test_yarn_off_leaves_the_tables_bit_identical():
    assert rope_scaling.yarn_parameters(TEXT) is None and rope_scaling.window(TEXT) == 262144
    inv, scale = rope_scaling.rope_tables(_cfg(None))
    before = (torch.tensor(1e7, dtype=torch.float64) ** (
        -torch.arange(0, 32, dtype=torch.float64) / 32)).to(torch.float32)      # the loader's expression before F9
    assert scale == 1.0 and torch.equal(inv, before)


def test_bad_factors_and_rope_types_refuse():
    for bad in (1.0, 0.5, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="above 1"):
            rope_scaling.yarn_parameters(TEXT, bad)
    linear = dict(TEXT, rope_parameters={"rope_type": "linear", "factor": 2.0, "rope_theta": 1e7})
    with pytest.raises(ValueError, match="rope_type"):
        rope_scaling.yarn_parameters(linear, 2.0)
    assert rope_scaling.yarn_parameters(linear) is None        # untouched without --rope-yarn, as before


def _flash_config(tmp_path, text=TEXT):
    full = {"hidden_size": 2048, "num_hidden_layers": 4, "layer_types": ["linear_attention", "full_attention"] * 2,
            "vocab_size": 1024, "rms_norm_eps": 1e-6, "num_attention_heads": 16, "num_key_value_heads": 2,
            "linear_num_key_heads": 16, "linear_num_value_heads": 32, "linear_key_head_dim": 128,
            "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4, "num_experts": 8, "num_experts_per_tok": 2,
            "moe_intermediate_size": 512, "shared_expert_intermediate_size": 512, "eos_token_id": 1, **text}
    (tmp_path / "config.json").write_text(json.dumps({"text_config": full}))


def test_config_reads_the_window_and_parameters(tmp_path):
    pytest.importorskip("triton")                    # the CUDA loader's module imports its kernels
    from tensorfold.families.qwen4_exp.cuda.weights import Config

    _flash_config(tmp_path)
    plain = Config.read(tmp_path)
    assert plain.yarn is None and plain.window == 262144 and plain.rotary_dim == 64
    on = Config.read(tmp_path, rope_yarn=2.0)
    assert on.yarn["factor"] == 2.0 and on.yarn["original_max_position_embeddings"] == 262144
    assert on.window == 524288 and on.mrope_section == (11, 11, 10)


def test_capacity_plan_window_doubles():
    geometry = Geometry(lambda slots: slots * 1024, 8)
    weights = CapWeights(10 * 2**30, 1 * 2**30)
    budget = 10 * 2**30 + 2 * 2**30 + 524288 * 1024 * 2
    native = make_plan(262144, 262144, False, budget, weights, geometry)
    doubled = make_plan(rope_scaling.window(TEXT, 2.0), 524288, False, budget, weights, geometry)
    assert choose(native) == 262144 and choose(doubled) == 524288 and doubled.native == 2 * native.native
    with pytest.raises(ValueError, match="native window"):           # plain RoPE still caps an explicit 524,288
        choose(make_plan(262144, 524288, True, budget, weights, geometry))


def test_cli_window_and_option(tmp_path, monkeypatch):
    from tensorfold.cuda import server

    _flash_config(tmp_path)
    assert cli._model_context(tmp_path) == 262144
    assert cli._model_context(tmp_path, 2.0) == 524288
    seen = []
    package = SimpleNamespace(cuda_engine=lambda *a, **kw: seen.append(kw) or SimpleNamespace(context_window=99))
    family = SimpleNamespace(title="Test", model_type="qwen4_exp", package=package)
    monkeypatch.setattr(server, "App", lambda *a, **kw: SimpleNamespace(effective_context_window=99))
    monkeypatch.setattr(server, "serve", lambda *a: None)
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--backend", "cuda", "--no-drafts",
                                          "--rope-yarn", "2"])
    cli._serve_cuda(args, family, tmp_path, 524288)
    assert seen[0]["rope_yarn"] == 2.0 and seen[0]["context"] == 524288
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--backend", "cuda", "--no-drafts"])
    cli._serve_cuda(args, family, tmp_path, 262144)
    assert "rope_yarn" not in seen[1]                                  # off: the engine is called as before


def test_serve_options_refuse_yarn_off_flash_next_cuda():
    from tensorfold.serve_options import check

    def args(factor):
        return cli.build_parser().parse_args(["serve", "m", "--rope-yarn", str(factor)])

    flash = SimpleNamespace(title="Flash Next", model_type="qwen4_exp", package=SimpleNamespace())
    other = SimpleNamespace(title="Other", model_type="qwen3_5", package=SimpleNamespace())
    check(args(2), flash, "cuda")
    with pytest.raises(ValueError, match="CUDA engine only"):
        check(args(2), flash, "mlx")
    with pytest.raises(ValueError, match="CUDA engine only"):
        check(args(2), other, "cuda")
    with pytest.raises(ValueError, match="above 1"):
        check(args(1), flash, "cuda")


def test_server_reports_the_engine_native_window(tmp_path):
    pytest.importorskip("tokenizers")
    pytest.importorskip("jinja2")
    from tokenizers import Tokenizer, models, pre_tokenizers

    from tensorfold.cuda import server

    _flash_config(tmp_path)
    tok = Tokenizer(models.WordLevel({"[UNK]": 0, "w": 1}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages }}"}))
    plain = server.App(SimpleNamespace(eos=(0,)), tmp_path, "test")
    assert plain.native_context_window == 262144
    yarn = server.App(SimpleNamespace(eos=(0,), native_window=524288), tmp_path, "test")
    assert yarn.native_context_window == 524288 and yarn.context_window == 524288


@pytest.mark.torch
@pytest.mark.parametrize("yarn,native", [(None, 65536), (2.0, 131072)])
def test_flash_engine_plans_the_yarn_window_before_loading(tmp_path, fake_runtime, capsys, yarn, native):  # noqa: F811
    from tests.test_cuda_capacity import HEAD, Loaded, small_config, checkpoint
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

    checkpoint(tmp_path, small_config(), HEAD)
    calls, _ = fake_runtime
    obj = FlashNextEngine.__new__(FlashNextEngine)
    with pytest.raises(Loaded):
        obj.__init__(tmp_path, master="example", max_len=None, context_explicit=False, depth=0, rope_yarn=yarn)
    assert obj.capacity_plan["native_window"] == native and obj.native_window == native
    assert calls[0]["rope_yarn"] == yarn
    out = capsys.readouterr().out
    assert ("[octojet] rope: YaRN factor 2.0, window 131072" in out) is (yarn is not None)
