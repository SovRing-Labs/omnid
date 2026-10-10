"""omnid v0 gates: ping, nomic parity with ollama, bitnet270m loads, concurrency, errors. Run: pytest -q projects/omnid/tests"""
import json, os, subprocess, sys, time, urllib.request, threading
from pathlib import Path
import numpy as np, pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from omnid_client import Omnid

SOCK = "/tmp/omnid-test.sock"


@pytest.fixture(scope="module")
def server():
    env = {**os.environ, "OMNID_SOCK": SOCK}
    p = subprocess.Popen([str(HERE / "omnid")], env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    for _ in range(100):
        if os.path.exists(SOCK): break
        time.sleep(0.05)
    yield
    p.terminate(); p.wait(10)


SAMPLES = ["The resonator settles on the residue that agrees with every ring.", "A shard is a mounted file of marks.",
           "Run the tests before claiming the task is done.", "The torus holds the state; the rings turn like gears.",
           "Phase vectors bind by addition modulo sixteen.", "The web face never binds to all interfaces.",
           "Embeddings are unit vectors with 768 dimensions.", "A claim session scopes every write to the task.",
           "Batching packs several sequences into one decode.", "The checksum slots refuse a tampered program.",
           "Search the field before judging what is possible.", "Every verified task is a mark the fleet learns from."]


def texts(n=12):
    d = Path(os.environ.get("OMNID_TEST_DOCS", "/nonexistent"))
    fs = sorted(d.glob("*/MANUAL.md"))[:n] if d.exists() else []
    return [" ".join(f.read_text(errors="ignore").split())[:1200] for f in fs] or SAMPLES[:n]


def test_ping(server):
    r = Omnid(SOCK).call(op="ping")
    assert r["server"] == "omnid" and "embed" in r["ops"] and "nomic" in r["models"]


def test_nomic_matches_ollama(server):
    T = texts()
    V = np.array(Omnid(SOCK).embed(T, kind="document"))
    try:
        O = []
        for t in T:
            req = urllib.request.Request("http://127.0.0.1:11434/api/embed", headers={"Content-Type": "application/json"},
                                         data=json.dumps({"model": "nomic-embed-text", "input": "search_document: " + t}).encode())
            v = np.array(json.load(urllib.request.urlopen(req, timeout=120))["embeddings"][0]); O.append(v / np.linalg.norm(v))
    except OSError:
        pytest.skip("ollama not reachable")
    cos = (V * np.array(O)).sum(1)
    assert V.shape[1] == 768 and cos.min() > 0.9999


def test_bitnet270m(server):
    v = Omnid(SOCK).embed(["the fleet dispatches a task"], model="bitnet270m", kind="raw")[0]
    assert len(v) == 640 and abs(np.linalg.norm(v) - 1) < 1e-4


def test_concurrent_clients(server):
    T = texts(4); ref = np.array(Omnid(SOCK).embed(T)); out = {}
    def work(i): out[i] = np.array(Omnid(SOCK).embed(T))
    th = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    [t.start() for t in th]; [t.join() for t in th]
    assert all(np.allclose(out[i], ref, atol=1e-6) for i in range(4))


def test_errors_do_not_kill_server(server):
    c = Omnid(SOCK)
    with pytest.raises(RuntimeError): c.call(op="nope")
    with pytest.raises(RuntimeError): c.call(op="embed", model="nope", texts=["x"])
    assert c.call(op="ping")["errors"] >= 2


# ---------------------------------------------------------------- worker forwarding + web face
def test_exec_forwarding(server):
    if not os.path.exists("/tmp/fleet-execd.sock"): pytest.skip("fleet-execd not running")
    r = Omnid(SOCK).call(op="exec", argv=["--help"], cwd=os.getcwd(), env={}, stdin="")
    assert r["code"] == 0 and "usage" in r["stdout"].lower()


def test_heads_forwarding(server):
    if not os.path.exists("/tmp/ring_headd.sock"): pytest.skip("ring_headd not running")
    assert "engine" in Omnid(SOCK).call(op="heads", health=True)["result"]


@pytest.fixture(scope="module")
def web(tmp_path_factory):
    cfg = tmp_path_factory.mktemp("cfg"); (cfg / ".config/omnid").mkdir(parents=True)
    up = tmp_path_factory.mktemp("up"); (up / "index.html").write_text("<script src=\"/assets/a.js\"></script>"); (up / "assets").mkdir()
    (up / "assets/a.js").write_text("console.log(1)")
    srv = subprocess.Popen([sys.executable, "-m", "http.server", "18931", "--bind", "127.0.0.1", "--directory", str(up)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    (cfg / ".config/omnid/web.json").write_text(json.dumps({"routes": [{"app": "demo", "port": 18931}]}))
    env = {**os.environ, "HOME": str(cfg), "OMNID_SOCK": "/tmp/omnid-webtest.sock", "OMNID_WEB": "127.0.0.1:18930,0.0.0.0:18932"}
    p = subprocess.Popen([str(HERE / "omnid")], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(1.5)
    yield (cfg / ".config/omnid/web.token").read_text().strip(), p
    p.terminate(); srv.terminate(); p.wait(10)


def get(path, token=None, ref=None, port=18930):
    h = {}
    if token: h["X-Omnid-Token"] = token
    if ref: h["Referer"] = ref
    try:
        r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}{path}", headers=h), timeout=10)
        return r.status, r.read(), r.headers
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers


def test_web_requires_token(web):
    assert get("/demo/")[0] == 401


def test_web_proxies_app_and_assets(web):
    tok, _ = web
    s, body, h = get("/demo/", tok); assert s == 200 and b"/assets/a.js" in body and h["X-Content-Type-Options"] == "nosniff"
    s, body, _ = get("/assets/a.js", tok, ref="http://127.0.0.1:18930/demo/"); assert s == 200 and body == b"console.log(1)"


def test_web_refuses_all_interfaces(web):
    with pytest.raises(OSError): urllib.request.urlopen("http://127.0.0.1:18932/", timeout=2)


def test_mcp_face(server):
    msgs = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26"}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "embed", "arguments": {"texts": ["hi"], "model": "nomic"}}},
            {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "nope", "arguments": {}}}]
    if os.path.exists("/tmp/fleet-execd.sock"):
        msgs.append({"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                     "params": {"name": "fleet_exec", "arguments": {"operations": [{"type": "bash", "command": "echo mcp-ok"}]}}})
    out = subprocess.run([str(HERE / "omnid"), "mcp"], input="\n".join(json.dumps(m) for m in msgs) + "\n", text=True,
                         capture_output=True, env={**os.environ, "OMNID_SOCK": SOCK}, timeout=300).stdout.splitlines()
    r = {j["id"]: j for j in map(json.loads, out)}
    assert r[1]["result"]["serverInfo"]["name"] == "omnid"
    names = {t["name"] for t in r[2]["result"]["tools"]}   # optional workers (shardd, arcd) list their tools only when present
    assert {"fleet_exec", "embed", "ring_heads"} <= names
    assert names <= {"fleet_exec", "embed", "ring_heads", "shard_search", "shard_put", "session_recall", "mark", "arc"}
    assert json.loads(r[3]["result"]["content"][0]["text"])["dim"] == 768
    assert "error" in r[4]
    if 5 in r: assert "mcp-ok" in r[5]["result"]["content"][0]["text"] and not r[5]["result"]["isError"]


def test_web_streams_sse(web, tmp_path):
    tok, _ = web
    srv = tmp_path / "sse.py"
    srv.write_text("import http.server,time\nclass H(http.server.BaseHTTPRequestHandler):\n def do_GET(s):\n  s.send_response(200);s.send_header('Content-Type','text/event-stream');s.end_headers()\n  for i in range(3):\n   s.wfile.write(f'data: {i}\\n\\n'.encode());s.wfile.flush();time.sleep(0.3)\nhttp.server.HTTPServer(('127.0.0.1',18931+2),H).serve_forever()\n")
    p = subprocess.Popen([sys.executable, str(srv)]); time.sleep(0.5)
    try:
        cfg = Path(os.environ.get("HOME"))  # routes are fixed per fixture; hit the SSE server through a fresh omnid
        home = tmp_path / "h"; (home / ".config/omnid").mkdir(parents=True)
        (home / ".config/omnid/web.json").write_text(json.dumps({"routes": [{"app": "sse", "port": 18933}]}))
        (home / ".config/omnid/web.token").write_text(tok + "\n")
        o = subprocess.Popen([str(HERE / "omnid")], env={**os.environ, "HOME": str(home), "OMNID_SOCK": "/tmp/omnid-sse.sock",
                             "OMNID_WEB": "127.0.0.1:18934"}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); time.sleep(1.5)
        r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:18934/sse/", headers={"X-Omnid-Token": tok, "Accept": "text/event-stream"}), timeout=10)
        t0 = time.time(); first = r.readline(); t_first = time.time() - t0
        rest = r.read().decode()
        assert first.startswith(b"data: 0") and t_first < 0.3 and "data: 2" in rest   # first event before the stream ends
        o.terminate()
    finally:
        p.terminate()
