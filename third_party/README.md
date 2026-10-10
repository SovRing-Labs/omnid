---
title: Third-party code used by omnid
updated: 2026-10-01
weaver_hash: 980d5406fadb71f3c991aeb867ed7bed503ae518283ee1babff6db76f9868945
weaver_imported: 2026-10-09T22:22:09+00:00
---
# Third-party code used by omnid

| Component | Version | Licence | Where |
|---|---|---|---|
| nlohmann/json (single header) | 3.11.2 | MIT (SPDX header in the file; © 2013-2022 Niels Lohmann) | `third_party/nlohmann/json.hpp` (vendored) |
| cpp-httplib | as vendored in llama.cpp (`vendor/cpp-httplib`) | MIT (© 2017 yhirose) | compiled from `$BITNET_DIR/3rdparty/llama.cpp/vendor/cpp-httplib` (not vendored here) |
| llama.cpp / ggml (via microsoft/BitNet `bitnet.cpp`) | the BitNet tree's llama.cpp | MIT | linked: `$BITNET_DIR/<build>/bin/libllama.so` |
| Models (not shipped) | nomic-embed-text v1.5 (Apache-2.0), BitNet-embedding-270M (see its model card) | per model | `$HOME/projects/bitnet/models/...` or `OMNID_NOMIC_GGUF` / `OMNID_BITNET_GGUF` |
