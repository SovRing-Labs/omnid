# omnid — one lightweight server for local embeddings, a hardened web face and an MCP face

<!-- status-badges (REPO-ACCESS-C) -->
[![status: pre-alpha (under development)](https://img.shields.io/badge/status-pre--alpha%20%28under%20development%29-orange)](STATUS.md) **Status: pre-alpha, under development** — what works today, what is untested and what failed: [`STATUS.md`](STATUS.md)

[![defensive publication: DOI 10.5281/zenodo.23282761](https://img.shields.io/badge/defensive%20publication-DOI%2010.5281%2Fzenodo.23282761-blue)](https://doi.org/10.5281/zenodo.23282761) **Defensive publication:** [`TECHNICAL-DISCLOSURE.md`](TECHNICAL-DISCLOSURE.md) · DOI [10.5281/zenodo.23282761](https://doi.org/10.5281/zenodo.23282761)
<!-- /status-badges -->

omnid is one small C++ binary that gives local agents, tools and web apps a single front door:

- **embed** — local text embeddings, no cloud: nomic-embed-text v1.5 (768-d; `search_document:` / `search_query:`
  prefixes, mean pooling) and BitNet-embedding-270M (640-d), on the llama.cpp engine from Microsoft's bitnet.cpp tree.
  Matches Ollama's nomic-embed-text to cosine > 0.9999 (see `TECHNICAL-DISCLOSURE.md` §1).
- **exec / heads** — forwards tool batches to a warm executor and queries to a head server over Unix sockets. Both are
  *optional* back ends you run yourself; omnid ships none of them.
- **web face** — one HTTP listener for many local web apps: path routing (`/<app>/…`), Referer fallback for SPA assets,
  one device token, security headers, SSE streaming; refuses to bind all interfaces.
- **MCP face** — `omnid mcp` is a stdio MCP server for opencode, Claude Code or any MCP client. It lists only the tools
  whose back end is actually running.

Protocol: newline-delimited JSON on `OMNID_SOCK` (default `/tmp/omnid.sock`), many requests per connection.

## Build

omnid links against the llama.cpp inside a [microsoft/BitNet](https://github.com/microsoft/BitNet) checkout (MIT) and
vendors `nlohmann/json` (MIT) under `third_party/`. Linux, g++ with C++17.

```bash
BITNET_DIR=~/projects/bitnet ./build.sh     # needs a built bitnet.cpp tree (libllama)
```

| var | meaning |
|---|---|
| `BITNET_DIR` | BitNet checkout (default `$HOME/projects/bitnet`) |
| `BN_BUILD` | build dir inside it (default `build-ring`) |
| `NLOHMANN_INC` | include dir for json (default: the vendored `third_party`) |
| `OMNID_NOMIC_GGUF`, `OMNID_BITNET_GGUF` | paths to the model files — **models are not shipped**; bring your own |

## Run

```bash
OMNID_WEB=127.0.0.1:8090 ./omnid                         # socket + web face
echo '{"op":"embed","model":"nomic","texts":["hello"]}' | ./omnid --once
./omnid mcp                                              # stdio MCP server (talks to the running omnid)
python3 -m pytest -q tests                               # needs numpy, pytest and the models; Ollama only for the parity test
```

Web routes: `~/.config/omnid/web.json` → `{"routes":[{"app":"notes","port":5173}]}`; the token is created at
`~/.config/omnid/web.token` on first start; open `/omnid/login?token=…` once per device.

## Optional workers

`exec`, `heads` and the memory/reactor ops are forwarded to separate processes you provide. They are off unless their
socket exists, and `tools/list` leaves out the tools they back.

| var | default | effect |
|---|---|---|
| `OMNID_DEFAULT_CWD` | `$HOME` | cwd for `exec` operations — set it to your own path |
| `OMNID_HOME` | the directory of the `omnid` binary | directory searched for an optional `shardd.py` worker |
| `OMNID_SHARDD` | on | `0` disables worker supervision |
| `OMNID_SHARDD_CMD` | `python3 $OMNID_HOME/shardd.py` | full command override; supervision starts only if the script exists or this is set |
| `SHARDD_SOCK`, `ARCD_SOCK` | `/tmp/omnid-shard.sock`, `/tmp/omnid-arc.sock` | worker sockets |

## Licence

**Apache License 2.0** — see `LICENSE` and `NOTICE`. This repository is Apache-2.0 throughout, including the text of
`TECHNICAL-DISCLOSURE.md`. Third-party code and models: `third_party/README.md`, `PROVENANCE.md`. Published as prior art;
no patents are filed.

## Dossier

History, specification, metrics, tests and limits: [`DOSSIER.md`](DOSSIER.md). The research record and essays: github.com/SovRing-Labs/omniring-research.
