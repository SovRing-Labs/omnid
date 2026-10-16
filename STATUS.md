> **Status: under development — pre-alpha.** Interfaces, numbers and file layout can change without notice.

# omnid — status

Last checked: 2026-10-16, Linux laptop (Intel i5-8365U, 8 threads), on a clean copy of this repo. 3 of 9 suite tests work today; 4 of 9 can run without special hardware. Each line below names its test id in `suite.json` and the evidence file it rests on.

## Works today

- **O1** Test suite is collectable: 12 tests present (no build needed) — PASS 2026-10-16 (`12 tests collected in 0.60s`); evidence: `tests/test_omnid.py`, `DOSSIER.md`.
- **O8** Hypervector wire: 100 % field decode on 4,506 real requests (measured elsewhere) — PASS 2026-10-16 (`found 2,560-byte`); evidence: `TECHNICAL-DISCLOSURE.md`, `DOSSIER.md`.
- **O9** Licence and provenance: Apache-2.0, no models or third-party sources shipped — PASS 2026-10-16 (`no model files at repo root`); evidence: `NOTICE`, `LICENSE`, `PROVENANCE.md`, `third_party/nlohmann/json.hpp`.

## Not yet tested

- **O2** Embeddings match Ollama's nomic-embed-text: cosine > 0.9999 — external: needs something the test laptop does not have; needs: a built microsoft/BitNet (bitnet.cpp) tree with libllama, g++ C++17, your own nomic-embed-text v1.5 GGUF, python3, pytest, numpy, Ollama serving nomic-embed-text on 127.0.0.1:11434
- **O3** BitNet-embedding-270M returns unit vectors of 640 dimensions — external: needs something the test laptop does not have; needs: a built microsoft/BitNet (bitnet.cpp) tree with libllama, g++ C++17, your own nomic-embed-text v1.5 GGUF, python3, pytest, numpy, BitNet-embedding-270M GGUF
- **O4** Web face: token required, proxy with security headers, no all-interface bind — external: needs something the test laptop does not have; needs: a built microsoft/BitNet (bitnet.cpp) tree with libllama, g++ C++17, your own nomic-embed-text v1.5 GGUF, python3, pytest, numpy
- **O5** SSE streams chunk by chunk through the web face — external: needs something the test laptop does not have; needs: a built microsoft/BitNet (bitnet.cpp) tree with libllama, g++ C++17, your own nomic-embed-text v1.5 GGUF, python3, pytest, numpy
- **O6** MCP face lists only tools whose back end runs, and embeds — external: needs something the test laptop does not have; needs: a built microsoft/BitNet (bitnet.cpp) tree with libllama, g++ C++17, your own nomic-embed-text v1.5 GGUF, python3, pytest, numpy
- Quick test `python3 -m pytest --collect-only -q -p no:cacheprovider tests` — not run on the laptop (long Rust or model build).

## Known failures and honest negatives

- **O7** Honest negative: q8_0 quantisation of nomic was slower (0.78x) and less exact. Unverifiable from this repo; a lead for you to re-measure, not a proven result. Recorded and re-checked 2026-10-16 (PASS); evidence: `TECHNICAL-DISCLOSURE.md`, `DOSSIER.md`
- Every behavioural test needs bitnet.cpp built and the author's GGUF model files; without them only the inventory row (O1) runs.

## Corrections

- **C12** — bitnet.cpp and upstream llama.cpp both build BitNet's FFN with SiLU while the model config says ReLU²; upstream perplexity not yet measured.

## How to re-check

- One file, any python3 >= 3.8: download [`sovring-suite.pyz`](https://github.com/SovRing-Labs/SovRing/releases/latest/download/sovring-suite.pyz), then `python3 sovring-suite.pyz --run omnid` (headless) or `python3 sovring-suite.pyz` (local page on 127.0.0.1).
- From this checkout: `bash install.sh` (checks python, prepares a venv, runs the suite headless, then opens the page; `--no-gui` to skip the page).
- Quick test only: `python3 -m pytest --collect-only -q -p no:cacheprovider tests`.

## Defensive publication

- `TECHNICAL-DISCLOSURE.md` — on public GitHub since 2026-10-10
- DOI: [10.5281/zenodo.23282761](https://doi.org/10.5281/zenodo.23282761) (Zenodo, see `CITATION.cff`)
- Licence: Apache-2.0 (LICENSE, NOTICE)
