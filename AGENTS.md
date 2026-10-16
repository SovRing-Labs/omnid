# AGENTS.md — omnid

For coding assistants (and people) setting this repository up. Plain steps; every command runs from the repo root.

**Status: pre-alpha, under development.** Interfaces, numbers and file layout can change without notice. Read `STATUS.md` first: what works today (each line names its test and evidence file), what is not yet tested, the known failures and the corrections.

## What it is
One small C++ server for local embeddings (nomic-embed-text v1.5 and BitNet-embedding-270M on llama.cpp), a hardened local web face and an MCP face. Models and the bitnet.cpp tree are not shipped.

## Install and check (3 commands)
```bash
git clone --depth 1 https://github.com/SovRing-Labs/omnid && cd omnid
python3 -m venv .venv && . .venv/bin/activate && python -m pip install -q numpy pytest
BITNET_DIR=~/projects/bitnet ./build.sh && python3 -m pytest -q tests   # expect: 0 failed
```
Needs: g++, bitnet.cpp build (libllama), embedding models, python3>=3.10, numpy, pytest.
Whole suite instead: `bash install.sh --no-gui` (python check → venv → every runnable `suite.json` test, headless; without `--no-gui` it then opens the local results page on 127.0.0.1).
Note: needs a built microsoft/BitNet tree and the nomic-embed / BitNet-embedding models first (README § Build); not a 5-minute test.

## What the tests mean
`suite.json` lists every claim this repository makes with its verify command, the expected output, what passing means, how the number was obtained and how to use it in your own work (9 tests, in `suite.json` at the repository root). `STATUS.md` says what works today, what is not yet tested and the known failures; the umbrella's `STATUS.md` has one row per repository.

## Where the evidence is
- `DOSSIER.md` — history, metrics and limits
- `TECHNICAL-DISCLOSURE.md` — the method, as dated prior art
- `CITATION.cff` — DOI 10.5281/zenodo.23282761

## Honest limits (from DOSSIER.md)
Neither the bitnet.cpp tree nor any model is shipped. The memory-search, reasoning and event workers are not in this repository; without them those ops are unavailable. Web-face behaviour under load is not measured.

## Licence
Apache-2.0.

## Guided setup through MCP
The SovRing umbrella ships an MCP server that explains, checks readiness, clones and runs these tests: https://github.com/SovRing-Labs/SovRing (README → "Paste this to your assistant"). Zero-install docs endpoint: https://gitmcp.io/SovRing-Labs/omnid
