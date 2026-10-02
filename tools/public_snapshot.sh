#!/usr/bin/env bash
# Build the public release commit of Octojet from a local ref, without the internal tree, and check it.
#
#   tools/public_snapshot.sh REF OUT [PUBLIC_CLONE]
#
# REF           local commit or tag to publish (e.g. v0.1.0)
# OUT           directory to create (must not exist); it becomes a git repo holding the release commit
# PUBLIC_CLONE  optional existing clone of the public repo: the release commit is added on top of its history
#               (later releases); without it OUT holds one root commit (the first release)
#
# Internal paths (EXCLUDE) never reach the public tree. engine/tools/release_check.py must pass on the result.
# Author: PUBLIC_AUTHOR_NAME / PUBLIC_AUTHOR_EMAIL (the public address), else the local repo's git identity. Nothing is pushed: push OUT's main yourself after review.
set -euo pipefail
EXCLUDE=(docs/superpowers docs/agent-brief.md)
[ $# -ge 2 ] || { echo "usage: $0 REF OUT [PUBLIC_CLONE]" >&2; exit 2; }
REF=$1; OUT=$2; PUB=${3:-}
ROOT=$(git rev-parse --show-toplevel)
[ ! -e "$OUT" ] || { echo "error: $OUT exists" >&2; exit 2; }
SHA=$(git -C "$ROOT" rev-parse --verify "$REF^{commit}")
VERSION=$(git -C "$ROOT" show "$SHA:engine/src/tensorfold/__init__.py" | sed -n 's/^__version__ = "\(.*\)"/\1/p')
mkdir -p "$OUT"
if [ -n "$PUB" ]; then
  cp -R "$PUB/.git" "$OUT/.git"
fi
git -C "$ROOT" archive "$SHA" | tar -x -C "$OUT"
for x in "${EXCLUDE[@]}"; do rm -rf "${OUT:?}/$x"; done
cd "$OUT"
[ -d .git ] || git init -q -b main
git add -A
python3 engine/tools/release_check.py
git -c user.name="${PUBLIC_AUTHOR_NAME:-$(git -C "$ROOT" config user.name)}" \
    -c user.email="${PUBLIC_AUTHOR_EMAIL:-$(git -C "$ROOT" config user.email)}" commit -q -m "${PUBLIC_MESSAGE:-Octojet ${VERSION}}" \
  -m "Release of Octojet ${VERSION}. Benchmarks with the exact versions compared: docs/benchmarks.md. Changes: engine/CHANGELOG.md."
echo "public release commit $(git rev-parse --short HEAD) (Octojet ${VERSION}) in $OUT: $(git ls-files | wc -l | tr -d ' ') files"
