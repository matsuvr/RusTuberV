#!/bin/sh
# Regenerate src/sys/bindings.rs from the vendored headers.
# Only maintainers run this; the output is checked in so consumers need no libclang.
set -eu
cd "$(dirname "$0")/.."

bindgen vendor/wrapper.h -o src/sys/bindings.rs \
  --dynamic-loading MpLib \
  --rust-target 1.85 --rust-edition 2024 --wrap-unsafe-ops \
  --allowlist-function 'Mp.*' \
  --allowlist-type 'Mp.*' \
  --allowlist-var 'Mp.*' \
  --default-enum-style rust_non_exhaustive \
  --raw-line '#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals, dead_code)]' \
  --raw-line '#![allow(clippy::undocumented_unsafe_blocks, clippy::cast_possible_truncation, clippy::semicolon_if_nothing_returned)]' \
  --raw-line '#![allow(missing_debug_implementations)]' \
  -- -x c++ -std=c++17 -Ivendor/headers

# bindgen's own formatting differs from rustfmt's, so without this the committed
# file (which `cargo fmt` has touched) never matches a fresh run, and CI's
# "bindings are up to date" diff check fails on formatting alone.
rustfmt --edition 2024 src/sys/bindings.rs

echo "regenerated from $(cat vendor/REF)"
