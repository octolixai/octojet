"""Octojet's Flash Next image/video port (upstream TensorFold v0.3.6.3 vision + MiaAI-Lab patches 0008/0009), checked
on the host: image prompts never meet F2d prefix reuse (same token ids, different images: no shared state), text keeps
its reuse and its exact call shapes, the engine routes images to the scheduler only, the options refuse what this fork
does not serve, and the CPU media positions follow Qwen's rope index with videos."""

import importlib
import sys
import threading
from types import SimpleNamespace as NS

import numpy as np
import pytest

from tensorfold.cuda.streams import Stream
from tests.test_cuda_geometry import allocations  # noqa: F401  (fixture: fake triton, so the modules import)
from tests.test_prefix_reuse_multi import decoder, finish


# -- the concurrent decoder (production's --parallel path) ------------------------------------------------------------

class Tower:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def encode(self, prepared, prompt):
        self.calls.append((prepared, tuple(prompt)))
        if self.fail:
            raise ValueError("image features do not match their placeholders")
        return NS(image=prepared, prompt=tuple(prompt))


def vision_decoder(monkeypatch, slots, log):
    torch = pytest.importorskip("torch")
    multi = importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")
    dec = decoder(multi, slots, monkeypatch, log)
    text_prefill = multi.prefill

    def prefill(e, prompt, sampling, *, mtp=True, resume=None, **image):
        if image:                                      # only image prompts pass the keyword (text calls unchanged)
            log.append(("image-prefill", tuple(prompt), resume, image["vision"].image))
        return text_prefill(e, prompt, sampling, mtp=mtp, resume=resume)

    monkeypatch.setattr(multi, "prefill", prefill)
    freed = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: freed.append(1))
    dec.vision = Tower()
    return dec, freed


def admit(dec, prompt, vision=None, draft=True):
    s = Stream(list(prompt), 4, None, draft=draft, vision=vision)
    dec.admit(s)
    return s


def test_an_image_prompt_never_reuses_a_text_entry_with_the_same_ids(allocations, monkeypatch):  # noqa: F811
    log = []
    dec, freed = vision_decoder(monkeypatch, 2, log)
    P = [1, 2, 3, 4]
    text = admit(dec, P)
    finish(dec, text)
    kept = list(dec.kept)
    assert len(kept) == 1
    image = object()
    s = admit(dec, P, vision=image)
    assert ("image-prefill", tuple(P), None, image) in log                   # a fresh prefill with its own features
    assert s.reuse is None and s.cached == 0 and s.reuse_miss is None
    assert dec.vision.calls == [(image, tuple(P))] and s.vision is None and freed
    assert dec.kept == kept and s.st is not text.st                          # not kept, the text entry untouched
    finish(dec, s)
    assert s.st in dec.free and dec.kept == kept


def test_two_image_prompts_with_the_same_ids_never_share_state(allocations, monkeypatch):  # noqa: F811
    log = []
    dec, _ = vision_decoder(monkeypatch, 2, log)
    P = [5, 6, 7]
    first, second = object(), object()
    a = admit(dec, P, vision=first)
    finish(dec, a)
    b = admit(dec, P, vision=second)
    finish(dec, b)
    prefills = [c for c in log if c[0] == "image-prefill"]
    assert [(c[2], c[3]) for c in prefills] == [(None, first), (None, second)]
    assert b.reuse is None and b.cached == 0 and dec.kept == []


def test_text_keeps_its_exact_hits_around_image_requests(allocations, monkeypatch):  # noqa: F811
    log = []
    dec, _ = vision_decoder(monkeypatch, 2, log)
    P = [1, 2, 3, 4]
    finish(dec, admit(dec, P))
    finish(dec, admit(dec, P, vision=object()))
    again = admit(dec, P)
    assert again.reuse == "exact" and again.cached == 4


def test_an_image_prompt_taking_an_idle_kept_slot_drops_its_entry(allocations, monkeypatch):  # noqa: F811
    log = []
    dec, _ = vision_decoder(monkeypatch, 1, log)
    P = [1, 2, 3, 4]
    text = admit(dec, P)
    finish(dec, text)
    s = admit(dec, P, vision=object())                     # no free slot: the idle kept one is overwritten
    assert s.st is text.st and dec.kept == []
    finish(dec, s)
    assert admit(dec, P).reuse is None                     # nothing stale points at the image's rows


@pytest.mark.parametrize("tower", ["missing", "fails"])
def test_an_image_prompt_that_cannot_encode_takes_no_slot(allocations, monkeypatch, tower):  # noqa: F811
    log = []
    dec, _ = vision_decoder(monkeypatch, 2, log)
    finish(dec, admit(dec, [1, 2, 3]))
    free, kept = list(dec.free), list(dec.kept)
    from tensorfold.server.errors import RequestError

    dec.vision = None if tower == "missing" else Tower(fail=True)
    with pytest.raises(RequestError):                      # the request's 400, not a server error
        admit(dec, [1, 2, 3], vision=object())
    assert dec.free == free and dec.kept == kept and dec.streams == {}
    assert not any(c[0] == "image-prefill" for c in log)


def test_serial_image_requests_run_too(allocations, monkeypatch):  # noqa: F811
    log = []
    dec, _ = vision_decoder(monkeypatch, 2, log)
    image = object()
    s = admit(dec, [1, 2, 3], vision=image, draft=False)
    assert ("image-prefill", (1, 2, 3), None, image) in log and dec.kept == [] and s.reuse is None


# -- the engine's routing -----------------------------------------------------------------------------------------------

def bare_engine(vision=True, scheduler=True):
    mod = importlib.import_module("tensorfold.families.qwen4_exp.cuda.engine")
    eng = mod.FlashNextEngine.__new__(mod.FlashNextEngine)
    eng.depth, eng.max_len, eng.tp = 3, 100, 1
    eng.vision = object() if vision else None
    eng.calls = []
    eng.scheduler = NS(submit=lambda *args, **kw: eng.calls.append((args, kw)) or {}) if scheduler else None
    return eng


def test_images_go_to_the_scheduler_and_text_calls_are_unchanged(allocations):  # noqa: F811
    pytest.importorskip("torch")
    eng, image, feed = bare_engine(), object(), (lambda ids: False)
    eng.generate([1, 2, 3], 5, None, feed)
    eng.generate([1, 2, 3], 5, None, feed, vision=image)
    (text_args, text_kw), (image_args, image_kw) = eng.calls
    assert "vision" not in text_kw and image_kw["vision"] is image
    assert text_args == image_args and {k: v for k, v in image_kw.items() if k != "vision"} == text_kw


@pytest.mark.parametrize("vision,scheduler", [(False, True), (True, False)])
def test_image_inputs_need_the_tower_and_the_scheduler(allocations, vision, scheduler):  # noqa: F811
    pytest.importorskip("torch")
    from tensorfold.server.errors import RequestError

    eng = bare_engine(vision, scheduler)
    with pytest.raises(RequestError, match="--vision"):
        eng.generate([1, 2, 3], 5, None, lambda ids: False, vision=object())
    assert eng.calls == []


def test_vision_workspace_setting(allocations, monkeypatch):  # noqa: F811
    pytest.importorskip("torch")
    mod = importlib.import_module("tensorfold.families.qwen4_exp.cuda.engine")
    monkeypatch.delenv("TENSORFOLD_VISION_WORKSPACE_MIB", raising=False)
    assert mod.vision_workspace() == mod.VISION_WORKSPACE
    monkeypatch.setenv("TENSORFOLD_VISION_WORKSPACE_MIB", "0")
    assert mod.vision_workspace() == 0
    monkeypatch.setenv("TENSORFOLD_VISION_WORKSPACE_MIB", "lots")
    with pytest.raises(ValueError):
        mod.vision_workspace()


def test_config_reads_the_rotary_sections(allocations):  # noqa: F811
    pytest.importorskip("torch")
    weights = importlib.import_module("tensorfold.families.qwen4_exp.cuda.weights")
    assert weights._sections([11, 11, 10]) == (11, 11, 10)
    for bad in ([11, 11], [11, -1, 10]):
        with pytest.raises(ValueError):
            weights._sections(bad)
    assert weights.Config.__dataclass_fields__["mrope_section"].default == (11, 11, 10)


# -- serve options and the CLI ------------------------------------------------------------------------------------------

def _config(tmp_path):
    import json

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen4_exp", "vision_config": {
        "out_hidden_size": 8}, "text_config": {"hidden_size": 8}}))
    return tmp_path


def _family(model_type="qwen4_exp"):
    return NS(model_type=model_type, title=model_type, package=NS(CUDA_KV_DTYPES=("bf16", "int8")))


def test_vision_options_are_checked_before_any_download(tmp_path):
    from tensorfold.serve_options import check, vision_options

    args = NS(vision=True, vision_urls=True, kv_dtype="int8", mtp_confidence=None)
    check(args, _family(), "cuda", _config(tmp_path))
    assert vision_options(args) == {"vision": True, "vision_urls": True}
    assert vision_options(NS(vision=False, vision_urls=False)) == {}
    with pytest.raises(ValueError, match="--vision-urls needs --vision"):
        check(NS(vision=False, vision_urls=True), _family(), "cuda")
    with pytest.raises(ValueError, match="CUDA engine"):
        check(NS(vision=True, kv_dtype="bf16"), _family(), "mlx", tmp_path)
    with pytest.raises(ValueError, match="Flash Next"):
        check(NS(vision=True, kv_dtype="bf16"), _family("qwen3_5"), "cuda", tmp_path)
    with pytest.raises(ValueError, match="vision"):
        check(NS(vision=True, kv_dtype="bf16"), _family("gemma4"), "cuda", tmp_path)


def test_cli_parses_the_vision_flags():
    from tensorfold.cli import build_parser

    args = build_parser().parse_args(["serve", "model", "--vision", "--vision-urls"])
    assert args.vision and args.vision_urls
    plain = build_parser().parse_args(["serve", "model"])
    assert not plain.vision and not plain.vision_urls


# -- the CUDA server: image requests prepared apart, text untouched -----------------------------------------------------

def _app(monkeypatch, prepared_vision):
    from tensorfold.cuda.server import App
    from tensorfold.server import prompts

    app = App.__new__(App)
    app.tok = NS(encode=lambda text, **kw: NS(ids=[21, 22]), decode=lambda tokens, **kw: "".join(map(chr, tokens)))
    app.vision = NS(frontend=NS(config={}))
    app.context_window, app.native_context_window, app.max_tokens, app.default_thinking = 64, 64, 8, False
    app.lock, app.served = threading.Lock(), "m"
    app.sampling_for = lambda *args: None
    app.renders, app.calls = [], []
    app.template = NS(render=lambda messages, **kw: app.renders.append(kw) or "rendered")

    def generate(prompt, max_tokens, sampling, on_tokens, draft=True, **kw):
        app.calls.append((list(prompt), kw))
        on_tokens([65])
        return {}

    app.engine = NS(generate=generate, eos=(0,), context_window=64)

    def prepare_images(frontend, messages, render, *, context_limit=None):
        render(messages)
        return prompts.RenderedPrompt([10, 11, 12], vision=prepared_vision)

    monkeypatch.setattr(prompts, "prepare_images", prepare_images)
    return app


def test_image_and_video_requests_carry_their_vision_and_text_requests_do_not(monkeypatch):
    prepared = NS(token_ids=(10, 11, 12))
    app = _app(monkeypatch, prepared)
    video = [{"role": "user", "content": [{"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AA"}},
                                          {"type": "text", "text": "what happens?"}]}]
    for messages, image in ((video, True), ([{"role": "user", "content": "hi"}], False)):
        body = {"messages": messages, "max_tokens": 2}
        req = app.prepare(body, True)
        app.run(body, True, lambda delta: True, prepared=req)
        prompt, kw = app.calls[-1]
        assert (req.vision is prepared, kw) == ((True, {"vision": prepared}) if image else (False, {}))
        assert prompt == ([10, 11, 12] if image else [21, 22])
    assert app.renders[0].get("allow_images") is True and "allow_images" not in app.renders[1]


def test_media_parts_without_vision_are_refused_as_before():
    from tensorfold.server.errors import RequestError
    from tensorfold.server.prompts import prepare_images

    video = [{"role": "user", "content": [{"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AA"}}]}]
    with pytest.raises(RequestError, match="--vision"):
        prepare_images(None, video, str)


# -- CPU media positions (patch 0008's media_positions, Qwen3.5's rope index with videos) -------------------------------

CONFIG = {"vision_config": {"spatial_merge_size": 2}, "image_token_id": 9, "video_token_id": 8,
          "vision_start_token_id": 6, "vision_end_token_id": 7}


def test_images_place_exactly_as_before():
    from tensorfold.vision.qwen_processing import image_positions, media_positions

    tokens = [1, 6, 9, 9, 9, 9, 9, 9, 7, 2, 3]
    positions, delta, spans = image_positions(tokens, [[1, 4, 6]], CONFIG)
    again, delta2, spans2, frames = media_positions(tokens, [[1, 4, 6]], None, CONFIG)
    assert np.array_equal(positions, again) and (delta, spans) == (delta2, spans2) and frames == ()
    assert spans == ((2, 8),)
    assert positions[:, 0, 2:8].tolist() == [[2] * 6, [2, 2, 2, 3, 3, 3], [2, 3, 4, 2, 3, 4]]
    assert positions[0, 0, 8] == 5 and delta == 5 - 8                       # past the image: max(h, w) on


def test_video_frame_groups_are_blocks_in_order():
    from tensorfold.vision.qwen_processing import media_positions

    # text, then two timestamped frame groups of a (2, 4, 4) video grid: 4 pads each, timestamp text between
    tokens = [1, 6, 8, 8, 8, 8, 7, 4, 6, 8, 8, 8, 8, 7, 2]
    positions, delta, spans, frames = media_positions(tokens, np.zeros((0, 3)), [[2, 4, 4]], CONFIG)
    assert spans == () and frames == ((2, 6), (9, 13))
    assert positions[:, 0, 2:6].tolist() == [[2] * 4, [2, 2, 3, 3], [2, 3, 2, 3]]
    assert positions[:, 0, 6].tolist() == [4, 4, 4] and positions[:, 0, 9].tolist() == [7, 7, 7]
    tail = positions[:, 0, 13:]
    assert (tail == tail[0]).all() and delta == int(positions.max()) + 1 - len(tokens)
    with pytest.raises(ValueError, match="Video"):
        media_positions(tokens, np.zeros((0, 3)), None, CONFIG)               # videos off: refused
    with pytest.raises(ValueError, match="video"):
        media_positions(tokens, np.zeros((0, 3)), [[1, 4, 4]], CONFIG)        # one group short


def test_video_parts_are_opt_in():
    from tensorfold.vision.images import ImageInputError, split_images
    from tensorfold.vision.videos import VideoSource

    messages = [{"role": "user", "content": [{"type": "text", "text": "a"},
                                             {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AA"}}]}]
    with pytest.raises(ImageInputError, match="audio and video are unsupported"):
        split_images(messages)
    template, sources = split_images(messages, allow_videos=True)
    assert template[0]["content"][1] == {"type": "video"} and sources == [VideoSource("data:video/mp4;base64,AA")]
    three = [{"role": "user", "content": [messages[0]["content"][1]] * 3}]
    with pytest.raises(ImageInputError, match="at most 2 videos"):
        split_images(three, allow_videos=True)
    remote = [{"role": "user", "content": [{"type": "video_url", "video_url": {"url": "https://x.test/v.mp4"}}]}]
    with pytest.raises(ImageInputError, match="--vision-urls"):
        split_images(remote, allow_videos=True)


def test_video_sampling_and_timestamps():
    from tensorfold.vision.videos import DEFAULT_VIDEO_LIMITS, VideoInput, sample_indices

    assert sample_indices(300, 30.0, DEFAULT_VIDEO_LIMITS).tolist() == np.linspace(0, 299, 20).round().astype(int).tolist()
    assert len(sample_indices(10 ** 6, 30.0, DEFAULT_VIDEO_LIMITS)) == DEFAULT_VIDEO_LIMITS.max_frames
    clip = VideoInput(np.zeros((3, 4, 4, 3), dtype=np.uint8), (0, 15, 30), 30.0)
    assert clip.timestamps(2) == [0.25, 1.0]                                  # an odd last frame repeats
    other = VideoInput(np.ones((3, 4, 4, 3), dtype=np.uint8), (0, 15, 30), 30.0)
    assert clip.content_hash != other.content_hash


def test_vision_modules_import_without_their_runtime_dependencies():
    """transformers (the tower, the processor) and PyAV (videos) load only when a --vision server uses them; checked in
    a fresh interpreter, so modules other tests imported do not count."""
    import os
    import subprocess

    code = ("import sys, tensorfold.vision.qwen_cuda, tensorfold.vision.videos, tensorfold.vision.qwen_processing, "
            "tensorfold.server.prompts, tensorfold.cuda.server; "
            "loaded = [m for m in ('transformers', 'av', 'PIL') if m in sys.modules]; "
            "assert not loaded, loaded")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    done = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr

# -- prompt chunk rows (MiaAI-Lab 0006: TENSORFOLD_PREFILL_ROWS) ---------------------------------------------------------

def test_prompt_chunk_rows_default_by_vision_and_follow_the_setting(monkeypatch):
    from tensorfold.cuda.geometry import indexed_prefill_rows

    monkeypatch.delenv("TENSORFOLD_PREFILL_ROWS", raising=False)
    assert indexed_prefill_rows() == 4096 and indexed_prefill_rows(vision=True) == 2048
    monkeypatch.setenv("TENSORFOLD_PREFILL_ROWS", "1024")
    assert indexed_prefill_rows() == indexed_prefill_rows(vision=True) == 1024
    for bad in ("128", "32768"):
        monkeypatch.setenv("TENSORFOLD_PREFILL_ROWS", bad)
        with pytest.raises(ValueError, match="256 to 16,384"):
            indexed_prefill_rows()
    monkeypatch.setenv("TENSORFOLD_PREFILL_ROWS", "1026")
    assert indexed_prefill_rows() == 1026
    with pytest.raises(ValueError, match="multiple of 4"):
        indexed_prefill_rows(vision=True)


def test_the_admission_counts_the_chunk_rows():
    from tensorfold.cuda.geometry import indexed_stream_geometry
    from tests.test_cuda_capacity import small_config

    text = small_config()
    two, four = (indexed_stream_geometry(text, 3, 7, 8, mtp=True, kv_bits=8, prefill_rows=r).needed(4096)
                 for r in (2048, 4096))
    assert four > two
    assert indexed_stream_geometry(text, 3, 7, 8, mtp=True, kv_bits=8, prefill_rows=6144).needed(4096) - four == \
        four - two                                             # linear in the rows
