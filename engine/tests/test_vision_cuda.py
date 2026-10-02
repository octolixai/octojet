"""CUDA image admission, checkpoint and transport contracts without accelerator runtimes.

From upstream TensorFold v0.3.6.3 tests/test_vision_cuda.py (MIT); the dense Qwen 27B engine's tests are left out
(Octojet serves --vision for Flash Next only), Flash Next's tower checks added (MiaAI-Lab patch 0008)."""

import ast
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tensorfold.vision.qwen_cuda import (EncodedVision, broadcast_encoded, capacity_geometry,
                                       checkpoint_vision, validate_encoded, weight_transform)


def _checkpoint(path, model_type="qwen3_5"):
    vision = {"model_type": model_type, "hidden_size": 8, "out_hidden_size": 8, "depth": 1,
              "patch_size": 2, "temporal_patch_size": 2, "spatial_merge_size": 2, "in_channels": 3,
              "intermediate_size": 12, "num_heads": 2, "num_position_embeddings": 4}
    config = {"model_type": model_type, "vision_config": vision, "text_config": {
        "hidden_size": 8, "head_dim": 8, "rope_parameters": {"mrope_interleaved": True,
        "mrope_section": [2, 1, 1], "partial_rotary_factor": 1}}}
    (path / "config.json").write_text(json.dumps(config))
    shapes = {"patch_embed.proj.weight": [8, 2, 2, 2, 3], "patch_embed.proj.bias": [8],
              "pos_embed.weight": [4, 8], "merger.norm.weight": [8], "merger.norm.bias": [8],
              "merger.linear_fc1.weight": [32, 32], "merger.linear_fc1.bias": [32],
              "merger.linear_fc2.weight": [8, 32], "merger.linear_fc2.bias": [8]}
    for part, shape in {"norm1": [8], "norm2": [8], "attn.qkv": [24, 8], "attn.proj": [8, 8],
                        "mlp.linear_fc1": [12, 8], "mlp.linear_fc2": [8, 12]}.items():
        shapes[f"blocks.0.{part}.weight"] = shape
        shapes[f"blocks.0.{part}.bias"] = [shape[0]]
    offset, entries = 0, {}
    for name, shape in shapes.items():
        size = int(np.prod(shape)) * 2
        entries["vision_tower." + name] = {"dtype": "BF16", "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    _write_tensors(path, entries, offset)
    return entries, offset


def _write_tensors(path, entries, size):
    raw = json.dumps(entries).encode()
    (path / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(size))


@pytest.mark.parametrize("model_type", ["qwen3_5", "qwen4_exp"])
def test_vision_headers_do_not_load_tensor_payloads(tmp_path, model_type):
    _, size = _checkpoint(tmp_path, model_type)
    config, resident = checkpoint_vision(tmp_path)
    assert config["out_hidden_size"] == 8
    assert resident == size


@pytest.mark.parametrize("damage", ["missing", "quantized", "range", "shape"])
def test_incomplete_or_incompatible_towers_refuse_before_loading(tmp_path, damage):
    entries, size = _checkpoint(tmp_path)
    key = "vision_tower.blocks.0.attn.qkv.weight"
    if damage == "missing":
        del entries[key]
    elif damage == "quantized":
        entries[key]["dtype"] = "U32"
    elif damage == "range":
        entries[key]["data_offsets"] = [size, size + 24 * 8 * 2]
    else:
        entries[key]["shape"] = [12, 16]
    _write_tensors(tmp_path, entries, size)
    with pytest.raises(ValueError):
        checkpoint_vision(tmp_path)


def test_every_placeholder_has_exactly_one_feature_and_position():
    prompt = [10, 99, 99, 99, 99, 11, 12]
    positions = [[0, 1, 1, 1, 1, 3, 4], [0, 1, 1, 2, 2, 3, 4], [0, 1, 2, 1, 2, 3, 4]]
    validate_encoded((1, 2, 3, 4), positions, -2, prompt, 99, (4, 8), 8)
    for rows, pos, delta, shape in [((0, 1, 2, 3), positions, -2, (4, 8)),
                                   ((1, 2, 3, 4), positions, 0, (4, 8)),
                                   ((1, 2, 3, 4), [positions[0]] * 2, -2, (4, 8)),
                                   ((1, 2, 3, 4), positions, -2, (3, 8))]:
        with pytest.raises(ValueError):
            validate_encoded(rows, pos, delta, prompt, 99, shape, 8)


def test_vision_memory_is_reserved_only_on_the_tower_rank(tmp_path):
    from tensorfold.cuda.capacity import Geometry

    _checkpoint(tmp_path)
    base = lambda text: Geometry(lambda slots: slots * 64, 8)
    zero = capacity_geometry(base, tmp_path, True, 0)({})
    one = capacity_geometry(base, tmp_path, True, 1)({})
    plain = capacity_geometry(base, tmp_path, False, 0)({})
    assert zero.needed(32) > one.needed(32) > plain.needed(32)
    original = lambda *args: (0, 0)
    info = {"shape": [8, 8], "dtype": "BF16"}
    assert weight_transform(original, True, 0)("vision_tower.x", info) == (128, 0)
    assert weight_transform(original, True, 1)("vision_tower.x", info) == (0, 0)


def test_tp_transports_features_and_negative_offset_bit_for_bit(monkeypatch):
    records, arrays = [], []
    rank = [0]
    def share(values, r, device):
        if r == 0:
            records.append(list(values))
            return list(values)
        return records.pop(0)
    def broadcast(value, source):
        if rank[0] == 0:
            arrays.append(value.copy())
        else:
            value[:] = arrays.pop(0)
    torch = SimpleNamespace(bfloat16=np.uint16, int32=np.int32,
                            empty=lambda shape, dtype, device: np.empty(shape, dtype=dtype))
    distributed = SimpleNamespace(broadcast=broadcast)
    torch.distributed = distributed
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", distributed)
    monkeypatch.setitem(sys.modules, "tensorfold.families.qwen3_5.cuda.decode_tp", SimpleNamespace(_share=share))
    features = np.array([[0, 65535], [1, 32768]], dtype=np.uint16)
    positions = np.array([[0, 1, 1, 2], [0, 1, 2, 3], [0, 1, 1, 2]], dtype=np.int32)
    outgoing = EncodedVision((1, 2), features, positions, -1)
    broadcast_encoded(outgoing, 0, "cpu", hidden=2, prompt_length=4)
    rank[0] = 1
    received = broadcast_encoded(None, 1, "cpu", hidden=2, prompt_length=4)
    assert received.rows == (1, 2) and received.rope_delta == -1
    np.testing.assert_array_equal(received.features, features)
    np.testing.assert_array_equal(received.positions, positions)


def test_a_meta_built_tower_matches_a_normally_built_one_in_the_installed_transformers():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    from tensorfold.vision.qwen_cuda import rotary_frequencies

    raw = {"depth": 1, "hidden_size": 32, "num_heads": 2, "intermediate_size": 64, "patch_size": 4,
           "spatial_merge_size": 2, "temporal_patch_size": 2, "in_channels": 3, "out_hidden_size": 32,
           "num_position_embeddings": 16, "deepstack_visual_indexes": []}
    config = Qwen3_5VisionConfig(**raw)
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    built = Qwen3_5VisionModel(config).eval()
    with torch.device("meta"):
        meta = Qwen3_5VisionModel(config)
    meta.load_state_dict(built.state_dict(), strict=True, assign=True)
    rotary_frequencies(meta.rotary_pos_emb, raw, "cpu")
    for name, buffer in built.rotary_pos_emb.named_buffers():
        assert torch.equal(dict(meta.rotary_pos_emb.named_buffers())[name], buffer)
    pixels = torch.randn(16, 3 * 2 * 4 * 4)
    grid = torch.tensor([[1, 4, 4]])
    with torch.inference_mode():
        want = built(pixels, grid_thw=grid, return_dict=True).pooler_output
        got = meta.eval()(pixels, grid_thw=grid, return_dict=True).pooler_output
    assert torch.equal(got, want)


def test_flash_next_placeholders_are_images_and_video_frames():
    """Patch 0008: a Flash Next prompt's media rows are its image and video placeholders, together."""
    prompt = [10, 99, 98, 98, 11]
    positions = [[0, 1, 2, 2, 3], [0, 1, 2, 2, 3], [0, 1, 2, 3, 4]]
    validate_encoded((1, 2, 3), positions, 0, prompt, frozenset({99, 98}), (3, 8), 8)
    with pytest.raises(ValueError, match="placeholder"):
        validate_encoded((1, 2, 3), positions, 0, prompt, 99, (3, 8), 8)   # videos off: frames are not media


def test_tower_workspace_reserve_follows_the_engine_setting(tmp_path):
    from tensorfold.cuda.capacity import Geometry

    _checkpoint(tmp_path, "qwen4_exp")
    base = lambda text: Geometry(lambda slots: slots * 64, 8)
    small = capacity_geometry(base, tmp_path, True, 0, 1024)({})
    plain = capacity_geometry(base, tmp_path, False, 0, 1024)({})
    assert small.needed(32) - plain.needed(32) == 1024
