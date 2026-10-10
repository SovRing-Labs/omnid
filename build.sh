#!/usr/bin/env sh
# Build omnid against the bitnet.cpp libllama (build-ring = patched pooled-embedding build, same as ring_headd).
set -e
cd "$(dirname "$0")"
BN="${BITNET_DIR:-$HOME/projects/bitnet}"
BB="${BN_BUILD:-build-ring}"
# JSON header: vendored third_party/ (release) or the fleet copy (dev)
NL="${NLOHMANN_INC:-$( [ -f third_party/nlohmann/json.hpp ] && echo third_party || echo ../omnitool/onnxruntime_include )}"
g++ -O3 -march=native -std=c++17 -Wall -Wno-sign-compare \
  -I"$BN/3rdparty/llama.cpp/include" -I"$BN/3rdparty/llama.cpp/ggml/include" -I"$NL" -I"$BN/3rdparty/llama.cpp/vendor/cpp-httplib" \
  omnid.cpp "$BN/3rdparty/llama.cpp/vendor/cpp-httplib/httplib.cpp" -o omnid.new -L"$BN/$BB/bin" -lllama -lggml -lggml-base -lpthread -Wl,-rpath,"$BN/$BB/bin"
mv -f omnid.new omnid
echo "built $(pwd)/omnid (libllama: $BB)"
