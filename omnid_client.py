"""omnid client — newline-delimited JSON over the omnid Unix socket (OMNID_SOCK, default /tmp/omnid.sock)."""
import json, os, socket

SOCK = os.environ.get("OMNID_SOCK", "/tmp/omnid.sock")


class Omnid:
    def __init__(self, path=SOCK, timeout=600.0):
        self.s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.s.settimeout(timeout)
        self.s.connect(path)
        self.f = self.s.makefile("rb")

    def call(self, **req):
        self.s.sendall((json.dumps(req) + "\n").encode())
        r = json.loads(self.f.readline())
        if not r.get("ok"):
            raise RuntimeError(r.get("error", "omnid error"))
        return r

    def embed(self, texts, model="nomic", kind="document"):
        return self.call(op="embed", model=model, kind=kind, texts=list(texts))["vectors"]

    def close(self):
        self.f.close(); self.s.close()
