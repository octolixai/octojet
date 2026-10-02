"""``--packed-cache`` parses into the serve options; Flash Next's engine factory forwards it to the engine."""

from pathlib import Path

from tensorfold import cli


def _parse(argv):
    return cli.build_parser().parse_args(argv)


def test_flag_default_and_values():
    assert _parse(["serve", "m"]).packed_cache is None
    assert _parse(["serve", "m", "--packed-cache", "off"]).packed_cache == "off"
    assert _parse(["serve", "m", "--packed-cache", "verify"]).packed_cache == "verify"
    assert _parse(["serve", "m", "--packed-cache", "/tmp/x"]).packed_cache == "/tmp/x"


def test_factory_forwards_packed_cache(monkeypatch, tmp_path):
    from tensorfold.families import qwen4_exp
    from tensorfold.families.qwen4_exp.cuda import engine as engine_module

    seen = {}

    class Fake:
        def __init__(self, model_dir, **kw):
            seen.update(kw)

    monkeypatch.setattr(engine_module, "FlashNextEngine", Fake)
    (tmp_path / "config.json").write_text("{}")
    qwen4_exp.cuda_engine(tmp_path, no_drafts=True, packed_cache="/tf/octojet-cache/packed")
    assert seen["packed_cache"] == "/tf/octojet-cache/packed"
    seen.clear()
    qwen4_exp.cuda_engine(tmp_path, no_drafts=True)
    assert seen["packed_cache"] is None
