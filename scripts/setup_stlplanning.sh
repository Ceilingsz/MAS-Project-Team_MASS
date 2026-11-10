#!/usr/bin/env bash
# Fetch the authors' STLPlanning code at a pinned commit into third_party/ (run from the repo root).
set -euo pipefail
DEST=third_party/STLPlanning
COMMIT=23f9551f63e86140fec622f14acc870ea7bd81bc
[ -d "$DEST/.git" ] || git clone https://github.com/sundw2014/STLPlanning "$DEST"
git -C "$DEST" fetch --quiet origin "$COMMIT" 2>/dev/null || true
git -C "$DEST" checkout --quiet "$COMMIT"
pip install gurobipy numpy
echo "STLPlanning ready at $DEST (commit $COMMIT)"
