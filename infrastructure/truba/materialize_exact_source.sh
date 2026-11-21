#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 6 ]]; then
  echo "usage: materialize_exact_source.sh BASE_SOURCE REMOTE_ROOT BUNDLE BASE_COMMIT COMMIT BUNDLE_REF" >&2
  exit 64
fi

base_source="$1"
remote_root="$2"
bundle="$3"
base_commit="$4"
expected_commit="$5"
bundle_ref="$6"

safe_path() {
  [[ "$1" =~ ^/[A-Za-z0-9._/-]+$ ]] && [[ "$1" != *"/../"* ]] && [[ "$1" != */.. ]]
}
safe_hash() { [[ "$1" =~ ^[0-9a-f]{40}$ ]]; }

safe_path "$base_source"
safe_path "$remote_root"
safe_path "$bundle"
safe_hash "$base_commit"
safe_hash "$expected_commit"
[[ "$bundle" =~ ^/tmp/mutaco-source-[A-Za-z0-9._-]+\.bundle$ ]]
[[ "$bundle_ref" =~ ^refs/heads/[A-Za-z0-9._/-]+$ ]]

cleanup_bundle() { rm -f -- "$bundle"; }
trap cleanup_bundle EXIT

[[ -f "$bundle" ]]
[[ ! -e "$remote_root" ]]
[[ "$(git -C "$base_source" rev-parse HEAD)" == "$base_commit" ]]
[[ -z "$(git -C "$base_source" status --porcelain --untracked-files=all)" ]]

mkdir -p -- "$(dirname "$remote_root")"
mkdir -- "$remote_root"
git clone --no-hardlinks "$base_source" "$remote_root/source-exact" >/dev/null
git -C "$remote_root/source-exact" fetch "$bundle" \
  "$bundle_ref:refs/remotes/mutaco-transfer/source" >/dev/null
git -C "$remote_root/source-exact" checkout --detach "$expected_commit" >/dev/null
[[ "$(git -C "$remote_root/source-exact" rev-parse HEAD)" == "$expected_commit" ]]
[[ -z "$(git -C "$remote_root/source-exact" status --porcelain --untracked-files=all)" ]]

printf '{"schema_version":"mutaco-truba-source-materialization-v1","source_root":"%s/source-exact","source_commit":"%s","status":"materialized_clean"}\n' \
  "$remote_root" "$expected_commit"
