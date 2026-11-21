#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 || ! "$1" =~ ^/tmp/mutaco-source-[A-Za-z0-9._-]+\.bundle$ ]]; then
  echo "refusing unsafe transfer-bundle cleanup target" >&2
  exit 64
fi
rm -f -- "$1"
