#!/usr/bin/env python3
"""Build an exact-token-count prompt manifest for the prefill arms (runs where ``tokenizers`` is installed: the container).

  python3 bench/prefill_prompt.py --tokenizer /tf/flashnext-nvfp4-mixed/tokenizer.json --tokens 128000 --seed 1 --tag m128k --out m128k.json

The text starts with a line unique to (tag, tokens, seed), so no two manifests share a prefix (the server's prompt cache
reuses only strict extensions of an earlier prompt; a shared prefix would make a "cold" run partly warm). The body is
engine/tools/prefill_cold.py's corpus, repeated with a salted separator until the ids reach the count, then trimmed.

Derived manifests (F2d prefix reuse) share a prefix with their source on purpose:

  --from P.json --take K --tail T --seed S --tag T'   P' = P[:K] + T fresh tokens (shares exactly K tokens with P)
  --from P.json --extend T --seed S --tag T''         P'' = P + T fresh tokens  (shares all of P)

The fresh tokens come from a salted line plus the corpus (repeated until T ids exist); for --take the first fresh token
is chosen to differ from P[K], so the common prefix is exactly K (asserted before writing). The manifest records
"derived_from" (the source sha256) and "shared" (the common-prefix length).
"""
import argparse, hashlib, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine" / "tools"))

def sha(ids):
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def load(path):
    """A manifest whose ids are a non-empty int list matching tokens and sha256 (ValueError otherwise)."""

    with open(path) as f:
        m = json.load(f)
    ids, tokens = (m.get("ids"), m.get("tokens")) if isinstance(m, dict) else (None, None)
    if not isinstance(m, dict) or type(tokens) is not int or tokens <= 0 or not isinstance(ids, list) or not ids \
            or not all(type(t) is int for t in ids) or len(ids) != tokens or m.get("sha256") != sha(ids):
        raise ValueError(f"{path} is not a valid manifest (ids / tokens / sha256)")
    return m


def encode_at_least(tok, text, base, tag, seed, salt, count):
    """Ids of ``text``, extended with salted separators and the corpus until at least ``count`` ids exist."""

    ids = tok.encode(text, add_special_tokens=False).ids
    while len(ids) < count:
        text += f"\n[{tag} part {len(ids)} seed {seed} salt {salt}]\n" + base
        ids = tok.encode(text, add_special_tokens=False).ids
    return ids


def fresh(tok, base, tag, seed, count, avoid=None):
    """``count`` fresh ids from a salted line plus the corpus; with ``avoid`` set, the first id differs from it (the
    salt varies the line's first character, so the search cannot fail on a tokenizer that maps characters to ids)."""

    for salt in range(64):
        lead = chr(ord("A") + salt % 26) if salt else "["
        text = f"{lead}{tag} tail seed {seed} salt {salt}]\n" + base
        ids = encode_at_least(tok, text, base, tag, seed, salt, count)
        if avoid is None or ids[0] != avoid:
            return ids[:count]
    raise ValueError("could not find a first fresh token that differs from the source at the cut")


def write(out, tag, ids, extra):
    m = {"tokens": len(ids), "ids": ids, "sha256": sha(ids), "tag": tag, **extra}
    Path(out).write_text(json.dumps(m))
    print(json.dumps({"tag": tag, "tokens": len(ids), "sha256": m["sha256"], "out": out}))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", help="tokenizer.json (required without --from; defaults to the source's with --from)")
    ap.add_argument("--tokens", type=int, help="exact token count of a fresh manifest")
    ap.add_argument("--from", dest="src", help="derive from this manifest (with --take K --tail T, or --extend T)")
    ap.add_argument("--take", type=int); ap.add_argument("--tail", type=int); ap.add_argument("--extend", type=int)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    def fail(msg):
        print(f"error: {msg}", file=sys.stderr)
        return 2

    if a.src is None:
        if a.tokenizer is None or a.tokens is None:
            return fail("--tokenizer and --tokens are required without --from")
        if a.take is not None or a.tail is not None or a.extend is not None:
            return fail("--take/--tail/--extend need --from")
        if a.tokens <= 0:
            return fail(f"--tokens must be positive, got {a.tokens}")
    else:
        if a.tokens is not None:
            return fail("--tokens and --from are exclusive")
        if a.extend is not None and (a.take is not None or a.tail is not None):
            return fail("--extend excludes --take/--tail")
        if a.extend is None and (a.take is None or a.tail is None):
            return fail("--from needs either --take K --tail T or --extend T")
    from tokenizers import Tokenizer
    from prefill_cold import corpus
    base = corpus()
    if a.src is None:
        tok = Tokenizer.from_file(a.tokenizer)
        text = f"[{a.tag} {a.tokens} seed {a.seed}]\n" + base
        ids = tok.encode(text, add_special_tokens=False).ids
        while len(ids) < a.tokens:
            text += f"\n[{a.tag} part {len(ids)} seed {a.seed}]\n" + base
            ids = tok.encode(text, add_special_tokens=False).ids
        write(a.out, a.tag, ids[:a.tokens], {"seed": a.seed, "source": "prefill_cold.corpus", "tokenizer": a.tokenizer})
        return 0
    try:
        src = load(a.src)
    except (OSError, ValueError) as e:
        return fail(str(e))
    tokenizer = a.tokenizer or src.get("tokenizer")
    if not tokenizer:
        return fail("the source manifest names no tokenizer; pass --tokenizer")
    tok = Tokenizer.from_file(tokenizer)
    if a.extend is not None:
        if a.extend <= 0:
            return fail(f"--extend must be positive, got {a.extend}")
        shared, head = len(src["ids"]), list(src["ids"])
        tail = fresh(tok, base, a.tag, a.seed, a.extend)
    else:
        if not 0 < a.take < len(src["ids"]):
            return fail(f"--take must be in 1..{len(src['ids']) - 1}, got {a.take}")
        if a.tail <= 0:
            return fail(f"--tail must be positive, got {a.tail}")
        shared, head = a.take, list(src["ids"][:a.take])
        try:
            tail = fresh(tok, base, a.tag, a.seed, a.tail, avoid=src["ids"][a.take])
        except ValueError as e:
            return fail(str(e))
    ids = head + tail
    n = 0
    while n < min(len(ids), len(src["ids"])) and ids[n] == src["ids"][n]:
        n += 1
    if n != shared:
        return fail(f"derived manifest shares {n} tokens with the source, expected {shared}")
    write(a.out, a.tag, ids, {"seed": a.seed, "source": src.get("source"), "tokenizer": tokenizer,
                              "derived_from": src["sha256"], "shared": shared})
    return 0


if __name__ == "__main__":
    sys.exit(main())
