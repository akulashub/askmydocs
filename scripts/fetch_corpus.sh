#!/usr/bin/env bash
# Re-create data/corpus/fastapi from a pinned FastAPI commit (MIT licensed).
# The golden set's evidence spans are verified against this exact snapshot, so bump
# the commit deliberately and re-run `pytest tests/test_golden_and_prompts.py`.
set -euo pipefail

COMMIT="${FASTAPI_COMMIT:-79cab5bed28b24d92c185d236feb33f5783dfeb3}"
DEST="$(cd "$(dirname "$0")/.." && pwd)/data/corpus/fastapi"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

git -C "$TMP" init -q
git -C "$TMP" remote add origin https://github.com/fastapi/fastapi.git
git -C "$TMP" sparse-checkout set docs/en/docs/tutorial docs/en/docs/advanced
git -C "$TMP" fetch -q --depth 1 --filter=blob:none origin "$COMMIT"
git -C "$TMP" checkout -q FETCH_HEAD

rm -rf "$DEST"
mkdir -p "$DEST"
cp -r "$TMP/docs/en/docs/tutorial" "$TMP/docs/en/docs/advanced" "$DEST/"
cp "$TMP/LICENSE" "$DEST/LICENSE"
echo "FastAPI docs @ $COMMIT -> $DEST ($(find "$DEST" -name '*.md' | wc -l) markdown files)"
