"""The cut-point names used in the prefill sources are exactly the recorder's frozen blocks, every block is cut
somewhere, host cuts never nest, and the engine configures the recorder."""

import re
from pathlib import Path

from tensorfold.cuda import prefill_timing as pt

SRC = Path(__file__).resolve().parents[1] / "src" / "tensorfold"
FILES = [SRC / "families/qwen4_exp/cuda/forward.py", SRC / "families/qwen4_exp/cuda/attention.py",
         SRC / "cuda/moe.py", SRC / "families/qwen4_exp/cuda/mtp.py", SRC / "families/qwen4_exp/cuda/decode.py",
         SRC / "families/qwen4_exp/cuda/multi.py"]
BEGIN = re.compile(r'TIMER\.begin\("([a-z_]+)"')
HOST = re.compile(r'TIMER\.host_end\("([a-z_]+)"')


def _used():
    begins, hosts = set(), set()
    for f in FILES:
        text = f.read_text()
        begins |= set(BEGIN.findall(text))
        hosts |= set(HOST.findall(text))
    return begins, hosts


def test_every_used_name_is_a_known_block():
    begins, hosts = _used()
    assert begins <= pt.DEVICE_BLOCKS, begins - pt.DEVICE_BLOCKS
    assert hosts <= pt.HOST_BLOCKS, hosts - pt.HOST_BLOCKS


def test_every_block_is_cut_somewhere():
    begins, hosts = _used()
    assert pt.DEVICE_BLOCKS <= begins, pt.DEVICE_BLOCKS - begins
    assert pt.HOST_BLOCKS <= hosts, pt.HOST_BLOCKS - hosts


def test_engine_configures_the_recorder_and_prefill_keeps_last_logits():
    engine = (SRC / "families/qwen4_exp/cuda/engine.py").read_text()
    assert "TIMER.configure(" in engine and "attention_layers=" in engine and "slots=" in engine
    decode = (SRC / "families/qwen4_exp/cuda/decode.py").read_text()
    assert "e.last_logits = last" in decode and "TIMER.chunk_rows[" in decode


def test_mtp_and_draft_phases_are_set_and_restored():
    mtp = (SRC / "families/qwen4_exp/cuda/mtp.py").read_text()
    decode = (SRC / "families/qwen4_exp/cuda/decode.py").read_text()
    engine = (SRC / "families/qwen4_exp/cuda/engine.py").read_text()
    assert 'if saved_phase == "main":' in mtp and 'TIMER.phase = "mtp"' in mtp and "finally" in mtp
    assert "saved_phase = TIMER.phase" in decode and 'TIMER.phase = "draft"' in decode
    assert 'TIMER.phase = "draft"' in decode and 'TIMER.begin("draft"' not in decode
    # the recorder is configured with the engine's own chunk rows (TENSORFOLD_PREFILL_ROWS, MiaAI-Lab 0006)
    assert "rows = PREFILL_ROWS if exl3 else indexed_prefill_rows(vision)" in engine
    assert "prefill_rows=rows, capacity=self.max_len" in engine


def test_every_file_pairs_its_begins_and_ends():
    for f in FILES:
        text = f.read_text()
        assert text.count("TIMER.begin(") == text.count("TIMER.end("), f.name
        assert text.count("TIMER.host_begin(") == text.count("TIMER.host_end("), f.name


def _function(text: str, name: str) -> str:
    """The source of top-level function ``name``: from its ``def`` to the next top-level statement."""

    start = text.index(f"\ndef {name}(")
    rest = text[start + 1:]
    m = re.search(r"\n(?=[^\s#])", rest)
    return rest[:m.start()] if m else rest


RESTORE = "TIMER.layer, TIMER.rows, TIMER.pos = saved_layer, saved_rows, saved_pos"


def test_mtp_and_draft_wrappers_restore_phase_and_context_in_finally():
    mtp = _function((SRC / "families/qwen4_exp/cuda/mtp.py").read_text(), "mtp_forward")
    draft = _function((SRC / "families/qwen4_exp/cuda/decode.py").read_text(), "draft")
    for body in (mtp, draft):
        assert "finally:" in body and "TIMER.phase = saved_phase" in body
        assert body.index("finally:") < body.index("TIMER.phase = saved_phase")
        for field in ("TIMER.layer", "TIMER.rows", "TIMER.pos"):
            assert field in body
        assert RESTORE in body and body.index("finally:") < body.index(RESTORE)
    assert "TIMER.layer, TIMER.rows, TIMER.pos = -1, len(next_tokens), st.mtp_len" in mtp
    assert "TIMER.pos = position + j" in draft
