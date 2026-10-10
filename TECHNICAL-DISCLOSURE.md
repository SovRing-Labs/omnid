# omnid — Technical Disclosure, Part 1 (2026-10-01; licence note 2026-10-09): front door, hardened web face, hypervector wire

Dated prior art. Everything below was built and measured in the author's workspace; the repository tests are in `tests/`.
Several numbers below carry no result-file path in this repository: they are marked **(no published measurement)** and must be given a
harness and result before release (see `NOTES.md`). Terms are the field's: an "embedding" is a dense vector of a text; a
"frame" is the fixed-size hypervector described in §3.

Author: Juan Carlos De Santiago, with the Sovereign Raptor fleet (AI-assisted). Dated by the release commit and Zenodo DOI.
Licence: Apache-2.0 (this file included).

## 1. One front door for a local AI system
A single resident process serves every local model/tool request over newline-delimited JSON on a Unix socket; handlers
are mounted behind one dispatch table: `embed` (in-process llama.cpp contexts, lazy-loaded, one per model), `exec`
(forwarded to a warm fork-server tool executor), `heads` (forwarded to an associative ring-head server). Faces onto the
same table: Unix socket, HTTP (web face), stdio MCP (thin client to the socket). Files: `omnid.cpp`.
- Consolidation target: 6 servers in 3 languages (~1.2 GB RSS) **(no published measurement)**.
- Embeddings equal Ollama's nomic-embed-text to cosine > 0.9999 on the same GGUF (`tests/test_omnid.py`, parity test; needs
  Ollama and the model).
- Negative result: q8_0 quantisation of this encoder was slower (0.78×) and less exact (min cosine 0.9972) than f16 on a
  4-core laptop CPU **(no published measurement)**.

## 2. Hardened web face for personal apps
Path routing to local upstreams with prefix stripping; un-prefixed SPA asset requests routed by the Referer's first path
segment, then by a last-app cookie; one device token at the door (cookie set by a one-time login URL); security headers
on every response; SSE passthrough streamed chunk by chunk (first event delivered before the stream ends); listeners
restricted to loopback and a private-overlay address, all-interface binds refused; listener retry until the overlay
address exists.

## 3. The hypervector wire
Every request may carry a 2,560-byte frame: role-bound, phase-quantised (16 phases), 4-bit-packed bundle of its fields
(tool, head, argument, …). Measured on 4,506 distinct real tool requests: 100% field decode, 0 collisions (harness `experiments/harness/hv0.py`,
result `experiments/results/hv0.result.json`, omni-ring repository v0.2.0); any node can route, deduplicate (frame
digest) and cluster requests without parsing (see omni-ring Technical Disclosure Part 3, §7).

## 4. Designs disclosed (designed, not measured)
- **DONE-RING**: an agent claims a task → the server opens a claim session with a task-scoped kit (write scope, allowed
  operations, guard level) → every write is ledgered at write time (task, agent, path, hash, bytes, op, time, project) and
  framed as a hypervector → each project keeps a shard of triangles (task—produced→path, task—by→agent,
  task—needs→task) → completion verification re-hashes the ledger instead of parsing agent prose → note dedup at
  write time by body hash.
- **BELT**: a batching scheduler for CPU inference: one admission queue with frame-digest caching; multi-sequence
  embedding batches sorted by length; continuous token-level batching for generation with chunked prefill; lane/thread
  counts chosen per CPU from measurement; priority classes (interactive > fleet > background); shared-prefix KV reuse.
- **Universal release**: one binary + plugin manifests (name, schema, handler kind {C ABI, Python worker, subprocess},
  shards, faces) so every tool ships as a plugin reachable from MCP, CLI, HTTP and shared memory.
- **Ring dialect**: handlers/routers written as array statements over hypervector types (bind, bundle, permute, sift,
  gap, fold), executed on integer kernels; programs exchanged as checksum-verified program vectors.

## 5. Optional back ends (in the source, not in this release)
`omnid.cpp` forwards further ops to optional worker processes over Unix sockets (a memory-search worker, a reasoning-engine
worker, and an event-recording door: the `mark` op, which records an app event — target, outcome, weight,
detail — into a local event store run by that worker; the socket ops are `shard_search`, `shard_put`,
`session_recall`, `mark` and `arc.*`). Their implementations read workspace-specific stores and are not part of this
repository; with no worker running the ops are unavailable and the MCP face omits the corresponding tools.

## 6. Open / unmeasured
Everything in §4; every item marked "no published measurement"; web-face behaviour under load (not measured).

## 7. Provenance (borrowed / ours)
Borrowed: Model Context Protocol (open spec); llama.cpp/ggml via bitnet.cpp (MIT, linked); cpp-httplib and nlohmann/json
(MIT); continuous batching (vLLM, llama.cpp) for BELT; OWASP header practice. Ours: the dispatch table, web face, MCP face,
the frame wire and the designs in §4. Clean-room: no third-party source was copied; see `PROVENANCE.md`.
