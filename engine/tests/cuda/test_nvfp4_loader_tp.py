"""``weights.load`` refuses ``tp=(rank, 2)`` on the NVFP4 mixed checkpoint before the packed-table cache is touched.

With ``OCTOJET_NVFP4_FLASHNEXT=<a mixed served directory>``: the refusal is a ValueError and neither
``checkpoint_key`` nor ``TableCache`` (which would fingerprint the shards, create the directory) runs.
"""

import os
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

MODEL = os.environ.get("OCTOJET_NVFP4_FLASHNEXT", "")
pytestmark = pytest.mark.skipif(not MODEL or not Path(MODEL).is_dir(),
                                reason="set OCTOJET_NVFP4_FLASHNEXT to a mixed NVFP4 Flash Next directory")


def test_tp_refused_before_cache(monkeypatch, tmp_path):
    from tensorfold.cuda import packed_cache as pc
    from tensorfold.families.qwen4_exp.cuda import weights

    def touched(*args, **kwargs):
        raise AssertionError("cache touched")

    monkeypatch.setattr(pc, "checkpoint_key", touched)
    monkeypatch.setattr(pc, "TableCache", touched)
    with pytest.raises(ValueError, match="one GPU"):
        weights.load(MODEL, "cuda", mtp=True, tp=(0, 2), packed_cache=str(tmp_path))
    assert not any(tmp_path.iterdir())
