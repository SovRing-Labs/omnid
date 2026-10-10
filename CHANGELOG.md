# omnid changelog

## 0.1.0 (2026-10-10)
First public release. One C++ binary: local embeddings (nomic-embed-text v1.5, BitNet-embedding-270M) on the llama.cpp
engine from a bitnet.cpp tree; `exec` and `heads` forwarding to optional back ends over Unix sockets; a hardened web face
(path routing, device token, security headers, SSE streaming, no all-interface binds); a stdio MCP face that lists only
the tools whose back end is running. Licence: Apache-2.0. Technical Disclosure (prior art) includes the DONE-RING and BELT
designs (designed, not measured).
- Needs a built bitnet.cpp tree and your own model files; neither is shipped.
- Tests: `tests/test_omnid.py` (needs the models; Ollama only for the parity test).
- Environment variables `OMNID_DEFAULT_CWD` and `OMNID_HOME` make the former workspace paths overridable.
