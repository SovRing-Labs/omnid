# omnid — Provenance

**Licence:** Apache-2.0 (`LICENSE`, `NOTICE`). **Disclosure:** published as dated prior art (`TECHNICAL-DISCLOSURE.md`); no patents filed.

## How this code was written
Written by Juan Carlos De Santiago with the Sovereign Raptor fleet (AI-assisted), from published interface specifications and our own
measurements. No third-party source file was copied or adapted. The optional workers that read workspace-specific stores are not in this
repository.

## Ideas this builds on (cited, not copied)
| Idea | Source |
|---|---|
| One tool protocol for agents | Model Context Protocol (open specification) |
| Continuous batching and priority admission (BELT design) | vLLM (Kwon et al., SOSP 2023); llama.cpp server batching |
| Web-face hardening headers | OWASP secure-headers practice |
| Hypervector frames on the wire | omni-ring Technical Disclosure Part 3 §7 (our own) |

## Used, not copied
| Component | Licence | Shipped? |
|---|---|---|
| llama.cpp / ggml (through a bitnet.cpp tree) | MIT | No — linked at build, from your own tree |
| cpp-httplib, nlohmann/json | MIT | headers, attributed in `NOTICE` |
| Model files (nomic-embed-text v1.5, BitNet-embedding-270M) | their own licences | No |

## Checks to run on the export tree (before release)
ScanCode with snippet matching and gitleaks on the exact export; confirm no workspace path remains.
