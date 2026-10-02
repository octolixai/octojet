"""engine.py needs triton, so the --tp 2 refusal on the NVFP4 mixed checkpoint is checked structurally."""

from pathlib import Path

from tensorfold.families.qwen4_exp.cuda import nvfp4

ENGINE = Path(__file__).resolve().parents[1] / "src/tensorfold/families/qwen4_exp/cuda/engine.py"
MESSAGE = "the NVFP4 mixed checkpoint runs on one GPU: start it without --tp 2"


def test_mixed_tp_refusal_precedes_nccl_and_admission(tmp_path):
    (tmp_path / nvfp4.MARKER).write_text("{}")
    assert nvfp4.is_mixed(tmp_path)
    source = ENGINE.read_text()
    guard = source.index("if tp == 2 and is_mixed(model_dir):")
    assert MESSAGE in source[guard:guard + 300]
    assert guard < source.index("NCCL(") and guard < source.index("admit(")
    assert guard < source.index("import torch")
