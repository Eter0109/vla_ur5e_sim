#!/usr/bin/env bash
# Download the exact public base checkpoint used for cold-start training.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DESTINATION="${1:-${ROOT}/.runtime/models/smolvla_base}"
REVISION="c83c3163b8ca9b7e67c509fffd9121e66cb96205"
EXPECTED_SHA256="7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb"

mkdir -p "${DESTINATION}"
hf download lerobot/smolvla_base --revision "${REVISION}" --local-dir "${DESTINATION}"
echo "${EXPECTED_SHA256}  ${DESTINATION}/model.safetensors" | sha256sum --check --strict
