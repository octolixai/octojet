"""Image prompt and cache boundaries exercised without accelerator imports or model weights.

From upstream TensorFold v0.3.6.3 tests/test_vision_server.py (MIT): the shared normalization, preparation and CUDA
App tests; the dense Qwen 27B engine, Mac scheduler and family-prefill tests are left out (not in this fork). Octojet's
Flash Next engine and prefix-reuse boundaries are in test_vision_flashnext.py."""

from __future__ import annotations

import base64
import io
import sys
import threading
from types import ModuleType, SimpleNamespace as NS

import numpy as np
import pytest

from tensorfold.cuda.server import App
from tensorfold.server.errors import RequestError
from tensorfold.server.messages import normalize_messages
from tensorfold.server.prompts import prepare_images, prepare_prompt


@pytest.fixture(autouse=True)
def block_accelerators(monkeypatch):
    for name in ("mlx", "mlx.core", "mlx.nn", "mlx_lm", "mlx_vlm", "torch", "triton"):
        monkeypatch.setitem(sys.modules, name, None)


def image_messages(color="red"):
    image = pytest.importorskip("PIL.Image")
    output = io.BytesIO()
    image.new("RGB", (2, 2), color).save(output, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()
    return [{"role": "user", "content": [
        {"type": "text", "text": "before"}, {"type": "image_url", "image_url": {"url": url}},
        {"type": "text", "text": "after"},
    ]}]


class Frontend:
    def __init__(self, tokens=(10, 11, 12, 13)):
        self.tokens = tokens
        self.calls = []

    def prepare(self, rendered, images, *, max_prompt_tokens):
        self.calls.append((rendered, images, max_prompt_tokens))
        if max_prompt_tokens is not None and len(self.tokens) > max_prompt_tokens:
            raise ValueError("expanded image prompt exceeds the context limit")
        return NS(token_ids=self.tokens, image_hashes=tuple(image.content_hash for image in images))


class Tokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return "rendered with image markers"

    def encode(self, text, **kwargs):
        return [21, 22] if not kwargs else NS(ids=[21, 22])

    def decode(self, tokens, **kwargs):
        return "".join(chr(token) for token in tokens)


def prompt_app(frontend):
    return NS(tokenizer=Tokenizer(), tokenizer_lock=threading.Lock(), vision=frontend, late_system="user",
              context_window=32, reasoning_effort="medium", render=lambda *args, **kwargs: ([1, 2, 3], 2))


def cuda_app(frontend):
    app = App.__new__(App)
    app.tok, app.vision = Tokenizer(), frontend
    app.context_window, app.native_context_window = 32, 64
    app.max_tokens, app.default_thinking = 8, False
    app.lock, app.served = threading.Lock(), "test-model"
    app.template_calls = []
    app.engine_calls = []

    def render(messages, **kwargs):
        normalized = normalize_messages(messages, allow_images=kwargs.get("allow_images", False))
        app.template_calls.append((normalized, kwargs))
        return "rendered prompt"

    def generate(prompt, max_tokens, sampling, on_tokens, draft=True, **kwargs):
        app.engine_calls.append((prompt, max_tokens, sampling, draft, kwargs))
        on_tokens([65])
        return {"cached": 0}

    app.template = NS(render=render)
    app.engine = NS(generate=generate, eos=(0,), context_window=32)
    app.sampling_for = lambda *args: None
    return app


def test_normalize_images_preserves_parts_and_instruction_order():
    messages = [{"role": "developer", "content": "first"}, {"role": "system", "content": "second"},
                *image_messages(), {"role": "developer", "content": "later"}]
    result = normalize_messages(messages, allow_images=True, late_system="user")
    assert result[0] == {"role": "system", "content": "first\n\nsecond"}
    assert result[1] == messages[2] and isinstance(result[1]["content"], list)
    assert result[2] == {"role": "user", "content": "later"}
    assert messages[0]["role"] == "developer"


@pytest.mark.parametrize("role", ["system", "developer", "assistant", "tool"])
def test_normalize_rejects_images_outside_user_role(role):
    messages = image_messages()
    messages[0]["role"] = role
    with pytest.raises(RequestError, match="only in user"):
        normalize_messages(messages, allow_images=True)


def test_normalize_images_are_opt_in_and_audio_remains_unsupported():
    with pytest.raises(RequestError, match="text parts only"):
        normalize_messages(image_messages())
    messages = image_messages()
    messages[0]["content"].append({"type": "input_audio", "input_audio": {"data": "x"}})
    with pytest.raises(RequestError, match="text and image_url"):
        normalize_messages(messages, allow_images=True)
    assert normalize_messages([{"role": "user", "content": [{"type": "text", "text": "a"},
                                                               {"type": "text", "text": "b"}]}],
                              allow_images=True)[0]["content"] == "ab"


def test_prepare_images_decodes_cpu_and_passes_expanded_tokens():
    frontend, templates = Frontend(), []
    prepared = prepare_images(frontend, image_messages(), lambda value: templates.append(value) or "rendered",
                              context_limit=8)
    assert prepared.tokens == [10, 11, 12, 13] and prepared.history_len == 0
    assert prepared.vision.image_hashes == (frontend.calls[0][1][0].content_hash,)
    assert templates[0][0]["content"][1] == {"type": "image", "detail": "auto"}
    assert frontend.calls[0][2] == 8 and frontend.calls[0][1][0].pixels == bytes([255, 0, 0]) * 4


def test_prepare_images_preserves_zero_capacity():
    with pytest.raises(RequestError, match="context limit"):
        prepare_images(Frontend(), image_messages(), str, context_limit=0)


def test_prepare_images_errors_are_request_refusals():
    with pytest.raises(RequestError, match="--vision"):
        prepare_images(None, image_messages(), str)
    messages = image_messages()
    messages[0]["content"][1]["image_url"]["url"] = "data:image/png;base64,aW52YWxpZA=="
    with pytest.raises(RequestError, match="invalid or unsupported"):
        prepare_images(Frontend(), messages, str)
    with pytest.raises(RequestError, match="context limit"):
        prepare_images(Frontend(), image_messages(), str, context_limit=3)


def test_prepare_prompt_preserves_text_render_and_direct_prompt_paths():
    app = prompt_app(None)
    prepared = prepare_prompt(app, [{"role": "user", "content": "text"}], [], False, None, {})
    assert (prepared.tokens, prepared.history_len, prepared.vision) == ([1, 2, 3], 2, None)
    assert prepare_prompt(app, None, [], False, "direct", {}).tokens == [21, 22]
    assert prepare_prompt(app, None, [], False, [3, 4], {}).tokens == [3, 4]
    assert not app.tokenizer.calls


def test_prepare_prompt_passes_tools_thinking_and_normalized_arguments():
    app = prompt_app(Frontend())
    calls = [{"function": {"name": "lookup", "arguments": '{"query":"x"}'}}]
    messages = [{"role": "assistant", "content": None, "tool_calls": calls}, *image_messages()]
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    prepared = prepare_prompt(app, messages, tools, True, None, {"reasoning_effort": "high"})
    template, kwargs = app.tokenizer.calls[0]
    assert prepared.vision is not None and prepared.history_len == 0
    assert kwargs == dict(add_generation_prompt=True, tokenize=False, enable_thinking=True, tools=tools,
                          reasoning_effort="high")
    assert template[0]["tool_calls"][0]["function"]["arguments"] == {"query": "x"}
    assert isinstance(calls[0]["function"]["arguments"], str)


def test_cuda_prepare_forwards_images_and_checks_expanded_context():
    app = cuda_app(Frontend())
    body = {"messages": image_messages(), "max_tokens": 2, "chat_template_kwargs": {"enable_thinking": True}}
    prepared = app.prepare(body, True)
    assert prepared.prompt == [10, 11, 12, 13] and prepared.vision is not None and prepared.thinking
    assert app.template_calls[0][1]["allow_images"] is True
    app.engine.context_window = 5
    with pytest.raises(RequestError, match="4 tokens.*2 reply tokens"):
        app.prepare(body, True)
    assert not app.engine_calls


def test_cuda_prepare_rejects_expanded_prompt_before_engine_submission():
    app = cuda_app(Frontend(tuple(range(40))))
    with pytest.raises(RequestError, match="expanded image prompt"):
        app.prepare({"messages": image_messages()}, True)
    assert not app.engine_calls


@pytest.mark.parametrize("image", [False, True])
@pytest.mark.parametrize("draft", [False, True])
def test_cuda_run_forwards_vision_without_changing_text_call_options(image, draft):
    app = cuda_app(Frontend())
    messages = image_messages() if image else [{"role": "user", "content": "text"}]
    body = {"messages": messages, "draft": draft, "max_tokens": 2}
    prepared = app.prepare(body, True)
    deltas = []
    result = app.run(body, True, lambda chunk: deltas.append(chunk) or True, prepared=prepared)
    prompt, count, sampling, received_draft, options = app.engine_calls[0]
    assert prompt == prepared.prompt and count == 2 and sampling is None and received_draft is draft
    assert options == ({"vision": prepared.vision} if image else {})
    assert result["content"] == "A" and deltas == [{"content": "A"}]


def test_prepare_images_fetches_urls_only_when_the_frontend_allows(monkeypatch):
    from tensorfold.vision import images

    messages = image_messages()
    part = messages[0]["content"][1]
    data = base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
    part["image_url"] = {"url": "https://example.com/image.png"}
    monkeypatch.setattr(images, "fetch_image", lambda *args, **kwargs: (data, "image/png"))
    with pytest.raises(RequestError, match="--vision-urls"):
        prepare_images(Frontend(), messages, str)
    frontend = Frontend()
    frontend.allow_urls = True
    assert prepare_images(frontend, messages, str).tokens == [10, 11, 12, 13]


def test_image_preparation_is_bounded_and_refuses_with_capacity_errors(monkeypatch):
    import threading

    from tensorfold.server import prompts
    from tensorfold.server.errors import CapacityError

    monkeypatch.setattr(prompts, "IMAGE_SLOTS", threading.BoundedSemaphore(1))
    monkeypatch.setattr(prompts, "IMAGE_WAITERS", threading.BoundedSemaphore(1))
    monkeypatch.setattr(prompts, "IMAGE_WAIT_S", 0.01)
    held = prompts.image_slot()
    with pytest.raises(CapacityError, match="busy"):          # a slot never frees within the wait
        prompts.image_slot()
    assert prompts.IMAGE_WAITERS.acquire(blocking=False)      # the waiter it took was given back
    with pytest.raises(CapacityError, match="queue is full"):  # every waiter place taken: refused at once
        prompts.image_slot()
    prompts.IMAGE_WAITERS.release()
    held.release()
    prompts.image_slot().release()


def test_cuda_required_call_continuation_keeps_the_images(monkeypatch):
    from tensorfold.vision import qwen_processing

    app = cuda_app(Frontend())
    app.vision.frontend = NS(config={"image_token_id": 7})
    grown = []
    monkeypatch.setattr(qwen_processing, "continued",
                        lambda prepared, ids, config: grown.append((prepared, list(ids), config)) or NS(ids=list(ids)))
    cuts = iter([(0, [1000])])
    app._call_gate = lambda prompt, tools: NS(cut=lambda new: next(cuts, None), observe=lambda token: None)
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}]
    body = {"messages": image_messages(), "max_tokens": 4, "tools": tools, "tool_choice": "required"}
    prepared = app.prepare(body, True)
    app.run(body, True, lambda chunk: True, prepared=prepared)
    first, second = app.engine_calls[0], app.engine_calls[1]
    assert first[0] == prepared.prompt and first[4] == {"vision": prepared.vision}
    assert second[0] == [*prepared.prompt, 1000] and second[4] == {"vision": NS(ids=second[0])}
    assert grown == [(prepared.vision, second[0], {"image_token_id": 7})]
