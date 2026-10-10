# omnid — Dossier

*History, specification, metrics, tests and limits. Companion to `TECHNICAL-DISCLOSURE.md` and `README.md`. The research record:
github.com/SovRing-Labs/omniring-research.*

## What it is

One small C++ binary that gives local agents, tools and web apps a single front door: local text embeddings with no cloud,
forwarding of tool batches and head queries to warm back ends over Unix sockets, a hardened web face for personal apps, and an MCP
face. In the stack it is the **front door** that the hands (Tarsus), the memory (omni-ring) and the heads sit behind. Stage (essay 07):
system → primitive.

## Creation history

| Date (2026) | Milestone |
|---|---|
| 09-30 – 10-01 | One resident process replaces a set of separate servers: embed, exec and heads behind one dispatch table; socket, HTTP and MCP faces onto the same table |
| 10-01 | The hypervector wire: every request may carry a 2,560-byte frame, so any node can route, deduplicate and cluster requests without parsing; the BELT and DONE-RING designs written down |
| 10-01 | The author's request behind BELT: *"vllm allows for batching, llama cpp does some cpu batching sparsely, can we build our own batching system into our server?"* |
| 10-02 | The shard worker supervised as a child, so shared memory and the mark door survive restarts |
| 10-08 | BELT v0 built as the pod-side core queue (`corehostd`): three priorities, slots refilled continuously, a digest cache (research corpus doc 38) |
| 10-09 | Released under Apache-2.0 (the author's decision of 10-09); workspace-specific workers left out of the public repository |

## Specification (summary)

* **Ops:** `embed` (nomic-embed-text v1.5, 768-d; BitNet-embedding-270M, 640-d; in-process llama.cpp contexts, lazy-loaded),
  `exec` (forwarded to a warm executor), `heads` (forwarded to a head server).
* **Faces:** newline-delimited JSON on a Unix socket; HTTP web face (path routing with prefix stripping, one device token, security
  headers, SSE streamed chunk by chunk, loopback and private-overlay listeners only); stdio MCP that lists only tools whose back end runs.
* **Designs disclosed, not built in this repository:** DONE-RING (claimed tasks with a write-time ledger), BELT generation batching,
  a universal plugin release, a ring dialect.

## Metrics

| Property | Result |
|---|---|
| Embeddings against Ollama's nomic-embed-text, same GGUF | cosine > 0.9999 (`tests/test_omnid.py`, parity test) |
| Hypervector wire on 4,506 real tool requests | 100 % field decode, 0 collisions (measured in the omni-ring Part 3 package) |
| q8_0 against f16 for this encoder on a 4-core laptop | slower (0.78×) and less exact (recorded; result file not in this repository) |

## Tests

`tests/test_omnid.py` — needs a built bitnet.cpp tree and your own model files; Ollama only for the parity test.

## Limits

* Neither the bitnet.cpp tree nor any model is shipped.
* The memory-search, reasoning and event workers are not in this repository; without them those ops are unavailable.
* Web-face behaviour under load is not measured.

## Lineage and credits

Model Context Protocol; llama.cpp / ggml via bitnet.cpp (MIT, linked); cpp-httplib and nlohmann/json (MIT); continuous batching
(vLLM, llama.cpp) for BELT; OWASP header practice. `PROVENANCE.md`. Updates: weekly, recorded here with dates.
