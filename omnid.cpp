// omnid — one lightweight multilingual server for the nervous system (OMNID, 2026-10-01).
// v0 = the chassis + the `embed` handler.
//
// Face (v0): Unix socket (OMNID_SOCK, default /tmp/omnid.sock), newline-delimited JSON, many requests per connection,
// one thread per connection. Handlers: ping, embed. Embedding engine = the bitnet.cpp libllama (build-ring, the patched
// build ring_headd links): nomic-embed-text v1.5 (mean pooling, search_document:/search_query: prefixes) and
// BitNet-embedding-270M (last pooling, as ring_headd). Models load lazily on first use and stay resident.
//
// Build: projects/omnid/build.sh
#include "llama.h"
#include "nlohmann/json.hpp"
#include "httplib.h"
#include <fstream>
#include <random>
#include <sstream>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <sys/select.h>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <csignal>
#include <cmath>
#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <set>
#include <thread>
#include <vector>
#include <iostream>
using json = nlohmann::json;

static std::atomic<bool> g_stop{false};
static void on_signal(int) { g_stop = true; }
static std::string home() { return getenv("HOME") ? getenv("HOME") : "/tmp"; }

// ------------------------------------------------------------------------------------------------ embed handler
struct ModelSpec { std::string gguf; enum llama_pooling_type pool; std::string doc_prefix, query_prefix; };
static std::map<std::string, ModelSpec> model_specs() {
    std::string bn = home() + "/projects/bitnet/models/";
    return {
        {"nomic", {getenv("OMNID_NOMIC_GGUF") ? getenv("OMNID_NOMIC_GGUF")
                   : bn + "nomic-embed-text-v1.5/nomic-embed-1.5-f16.gguf",
                   LLAMA_POOLING_TYPE_MEAN, "search_document: ", "search_query: "}},
        {"bitnet270m", {getenv("OMNID_BITNET_GGUF") ? getenv("OMNID_BITNET_GGUF")
                   : bn + "bitnet-embedding-270m/bitnet-embeddings-270m-bf16-i2_s.gguf",
                   LLAMA_POOLING_TYPE_LAST, "", ""}},
    };
}

struct Embedder {
    llama_model *model = nullptr; llama_context *ctx = nullptr; const llama_vocab *vocab = nullptr;
    int n_embd = 0, n_ub = 2048; ModelSpec spec; std::mutex mu;
    Embedder(const ModelSpec &s, int threads) : spec(s) {
        auto mp = llama_model_default_params(); mp.n_gpu_layers = 0;
        model = llama_model_load_from_file(s.gguf.c_str(), mp);
        if (!model) throw std::runtime_error("cannot load " + s.gguf);
        auto cp = llama_context_default_params();
        cp.embeddings = true; cp.pooling_type = s.pool;
        n_ub = getenv("OMNID_UBATCH") ? std::atoi(getenv("OMNID_UBATCH")) : 2048;
        cp.n_ctx = n_ub; cp.n_batch = n_ub; cp.n_ubatch = n_ub; cp.n_threads = threads; cp.n_threads_batch = threads;
        ctx = llama_init_from_model(model, cp);
        if (!ctx) throw std::runtime_error("cannot create context");
        vocab = llama_model_get_vocab(model); n_embd = llama_model_n_embd(model);
    }
    ~Embedder() { if (ctx) llama_free(ctx); if (model) llama_model_free(model); }
    std::vector<float> embed(const std::string &text) {
        std::lock_guard<std::mutex> lk(mu);
        std::vector<llama_token> tok(text.size() + 16);
        int nt = llama_tokenize(vocab, text.c_str(), (int)text.size(), tok.data(), (int)tok.size(), true, true);
        if (nt < 0) { tok.resize(-nt); nt = llama_tokenize(vocab, text.c_str(), (int)text.size(), tok.data(), (int)tok.size(), true, true); }
        if (nt > n_ub) nt = n_ub;
        if (nt < 1) throw std::runtime_error("empty text");
        tok.resize(nt);
        llama_batch b = llama_batch_init(nt, 0, 1);
        for (int i = 0; i < nt; i++) { b.token[i] = tok[i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1; }
        b.n_tokens = nt;
        llama_memory_clear(llama_get_memory(ctx), true);
        if (llama_decode(ctx, b) < 0) { llama_batch_free(b); throw std::runtime_error("decode failed"); }
        const float *e = llama_get_embeddings_seq(ctx, 0);
        std::vector<float> v(e, e + n_embd);
        llama_batch_free(b);
        double ss = 0; for (float x : v) ss += (double)x * x;
        float norm = ss > 0 ? (float)(1.0 / std::sqrt(ss)) : 0.f;
        for (auto &x : v) x *= norm;
        return v;
    }
};

struct EmbedHandler {
    std::map<std::string, ModelSpec> specs = model_specs();
    std::map<std::string, std::unique_ptr<Embedder>> loaded; std::mutex mu; int threads;
    explicit EmbedHandler(int t) : threads(t) {}
    Embedder &get(const std::string &name) {
        std::lock_guard<std::mutex> lk(mu);
        auto it = loaded.find(name);
        if (it != loaded.end()) return *it->second;
        auto s = specs.find(name);
        if (s == specs.end()) throw std::runtime_error("unknown model: " + name);
        auto e = std::make_unique<Embedder>(s->second, threads);
        auto &ref = *e; loaded[name] = std::move(e);
        return ref;
    }
    json call(const json &req) {
        std::string name = req.value("model", "nomic");
        std::string kind = req.value("kind", "document");          // document | query | raw
        Embedder &e = get(name);
        std::string prefix = kind == "document" ? e.spec.doc_prefix : kind == "query" ? e.spec.query_prefix : "";
        json out = json::array();
        auto t0 = std::chrono::steady_clock::now();
        for (const auto &t : req.at("texts")) out.push_back(e.embed(prefix + t.get<std::string>()));
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        return {{"ok", true}, {"model", name}, {"dim", e.n_embd}, {"vectors", out}, {"ms", ms}};
    }
};


// ------------------------------------------------------------------------------------------------ web face (WEB-0)
// One HTTP listener per address in OMNID_WEB ("127.0.0.1:8090,100.x.y.z:8090"; never 0.0.0.0 unless OMNID_WEB_ALLOW_ANY).
// Path routing: /<app>/... -> upstream (prefix stripped). Requests without an app prefix (absolute /assets/.. from an SPA)
// are routed by the Referer's first path segment, then by the last app cookie. One token at the door (cookie omnid_token,
// set by /omnid/login?token=...), security headers on every response. Routes: ~/.config/omnid/web.json.
// SSE (text/event-stream) is streamed chunk by chunk. Known gap (v0): no WebSocket proxying.
static json forward_unix(const std::string &path, const json &payload, bool newline_terminated);
static std::string env_or(const char *k, const char *d);
struct WebRoute { std::string app, host; int port; };
struct WebFace {
    std::vector<WebRoute> routes; std::string token; std::atomic<long> hits{0}, denied{0}, upstream_err{0};
    std::vector<std::unique_ptr<httplib::Server>> servers;
    static std::string cfg_dir() { return home() + "/.config/omnid"; }
    void load() {
        std::ifstream f(cfg_dir() + "/web.json");
        if (!f) throw std::runtime_error("missing " + cfg_dir() + "/web.json");
        json j = json::parse(f);
        for (auto &r : j.at("routes")) routes.push_back({r.at("app"), r.value("host", "127.0.0.1"), r.at("port")});
        std::string tp = cfg_dir() + "/web.token";
        std::ifstream t(tp);
        if (t) std::getline(t, token);
        if (token.size() < 32) {
            std::random_device rd; std::stringstream ss;
            for (int i = 0; i < 32; i++) ss << std::hex << (rd() & 0xf);
            token = ss.str();
            std::ofstream o(tp); o << token << "\n"; o.close(); chmod(tp.c_str(), 0600);
        }
    }
    static std::string cookie(const httplib::Request &q, const std::string &name) {
        auto c = q.get_header_value("Cookie"); std::string k = name + "=";
        size_t p = c.find(k); if (p == std::string::npos) return "";
        size_t e = c.find(';', p); return c.substr(p + k.size(), e == std::string::npos ? std::string::npos : e - p - k.size());
    }
    const WebRoute *route_for(const std::string &seg) const {
        for (auto &r : routes) { if (r.app == seg) return &r; }
        return nullptr;
    }
    static std::string first_seg(const std::string &path) {
        size_t a = path.find_first_not_of('/'); if (a == std::string::npos) return "";
        size_t b = path.find('/', a); return path.substr(a, b == std::string::npos ? std::string::npos : b - a);
    }
    static void harden(httplib::Response &r) {
        r.set_header("X-Content-Type-Options", "nosniff");
        r.set_header("Referrer-Policy", "same-origin");
        r.set_header("X-Frame-Options", "SAMEORIGIN");
        r.set_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()");
    }
    void handle(const httplib::Request &q, httplib::Response &r) {
        hits++; harden(r);
        if (q.path == "/omnid/health") {
            json h = {{"ok", true}, {"hits", hits.load()}, {"denied", denied.load()}, {"upstream_errors", upstream_err.load()}};
            json apps = json::array(); for (auto &x : routes) apps.push_back(x.app); h["apps"] = apps;
            r.set_content(h.dump(), "application/json"); return;
        }
        if (q.path == "/omnid/login") {
            if (q.get_param_value("token") == token) {
                r.set_header("Set-Cookie", "omnid_token=" + token + "; Path=/; HttpOnly; SameSite=Strict; Max-Age=31536000");
                r.set_redirect("/"); return;
            }
            denied++; r.status = 403; r.set_content("bad token", "text/plain"); return;
        }
        if (cookie(q, "omnid_token") != token && q.get_header_value("X-Omnid-Token") != token) {
            denied++; r.status = 401; r.set_content("omnid: open /omnid/login?token=<token> once on this device", "text/plain"); return;
        }
        // the ops door for browser apps (same token): POST /omnid/api/<op> {json} -> the socket op.
        if (q.path.rfind("/omnid/api/", 0) == 0) {
            std::string op = q.path.substr(11);
            static const std::set<std::string> allowed = {"ping", "mark", "shard_search", "heads"};
            if (!allowed.count(op) || (q.method != "POST" && op != "ping")) { r.status = 404; r.set_content("{\"ok\":false,\"error\":\"no such op\"}", "application/json"); return; }
            json req = json::object();
            if (!q.body.empty()) { try { req = json::parse(q.body); } catch (...) { r.status = 400; r.set_content("{\"ok\":false,\"error\":\"bad json\"}", "application/json"); return; } }
            if (!req.is_object()) { r.status = 400; r.set_content("{\"ok\":false,\"error\":\"object required\"}", "application/json"); return; }
            req["op"] = op;
            json out; try { out = forward_unix(env_or("OMNID_SOCK", "/tmp/omnid.sock"), req, true); }
            catch (const std::exception &ex) { upstream_err++; out = {{"ok", false}, {"error", ex.what()}}; r.status = 502; }
            r.set_content(out.dump(), "application/json"); return;
        }
        std::string seg = first_seg(q.path), upath = q.path;
        const WebRoute *rt = route_for(seg);
        if (rt) { upath = q.path.substr(1 + seg.size()); if (upath.empty()) upath = "/"; }
        else {
            auto ref = q.get_header_value("Referer"); size_t p = ref.find("://");
            if (p != std::string::npos) { size_t s2 = ref.find('/', p + 3); if (s2 != std::string::npos) rt = route_for(first_seg(ref.substr(s2))); }
            if (!rt) rt = route_for(cookie(q, "omnid_app"));
        }
        if (!rt) {
            if (q.path == "/") {
                std::string h = "<!doctype html><title>omnid</title><h1>omnid</h1><ul>";
                for (auto &x : routes) h += "<li><a href=\"/" + x.app + "/\">" + x.app + "</a></li>";
                r.set_content(h + "</ul>", "text/html"); return;
            }
            r.status = 404; r.set_content("omnid: no app for " + q.path, "text/plain"); return;
        }
        std::string target = upath;
        size_t qm = q.target.find('?'); if (qm != std::string::npos) target += q.target.substr(qm);   // raw query, untouched
        httplib::Client c(rt->host, rt->port); c.set_connection_timeout(5); c.set_read_timeout(120);
        httplib::Headers hd;
        for (auto &h : q.headers) {
            std::string k = h.first; std::string kl = k; for (auto &ch : kl) ch = tolower(ch);
            if (kl == "host" || kl == "cookie" || kl == "content-length" || kl == "connection" || kl == "accept-encoding") continue;
            hd.emplace(k, h.second);
        }
        std::string ct = q.get_header_value("Content-Type");
        // SSE (EventSource sends Accept: text/event-stream): stream upstream chunks as they arrive instead of buffering.
        if (q.method == "GET" && q.get_header_value("Accept").find("text/event-stream") != std::string::npos) {
            WebRoute up = *rt; httplib::Headers hd2 = hd; std::string tg = target;
            r.set_header("Cache-Control", "no-cache");
            r.set_header("Set-Cookie", "omnid_app=" + up.app + "; Path=/; SameSite=Strict");
            r.set_chunked_content_provider("text/event-stream", [this, up, hd2, tg](size_t, httplib::DataSink &sink) {
                httplib::Client sc(up.host, up.port); sc.set_connection_timeout(5); sc.set_read_timeout(24 * 3600);
                auto res2 = sc.Get(tg, hd2, [&](const char *d, size_t n) { return sink.write(d, n); });
                if (!res2) upstream_err++;
                sink.done(); return true;
            });
            return;
        }
        httplib::Result res(nullptr, httplib::Error::Unknown);
        if (q.method == "GET") res = c.Get(target, hd);
        else if (q.method == "HEAD") res = c.Head(target, hd);
        else if (q.method == "POST") res = c.Post(target, hd, q.body, ct);
        else if (q.method == "PUT") res = c.Put(target, hd, q.body, ct);
        else if (q.method == "PATCH") res = c.Patch(target, hd, q.body, ct);
        else if (q.method == "DELETE") res = c.Delete(target, hd, q.body, ct);
        else if (q.method == "OPTIONS") res = c.Options(target, hd);
        if (!res) { upstream_err++; r.status = 502; r.set_content("omnid: upstream " + rt->app + " unreachable", "text/plain"); return; }
        r.status = res->status;
        for (auto &h : res->headers) {
            std::string kl = h.first; for (auto &ch : kl) ch = tolower(ch);
            if (kl == "content-length" || kl == "transfer-encoding" || kl == "connection" || kl == "content-type") continue;
            if (kl == "location" && !h.second.empty() && h.second[0] == '/') { r.set_header(h.first, "/" + rt->app + h.second); continue; }
            r.set_header(h.first, h.second);
        }
        r.set_header("Set-Cookie", "omnid_app=" + rt->app + "; Path=/; SameSite=Strict");
        r.set_content(res->body, res->get_header_value("Content-Type").empty() ? "application/octet-stream" : res->get_header_value("Content-Type"));
    }
    void start(const std::string &spec) {
        load();
        std::stringstream ss(spec); std::string item;
        while (std::getline(ss, item, ',')) {
            size_t c = item.rfind(':'); std::string host = item.substr(0, c); int port = std::atoi(item.substr(c + 1).c_str());
            if ((host == "0.0.0.0" || host == "::") && !getenv("OMNID_WEB_ALLOW_ANY")) { std::cerr << "[omnid] refusing to bind " << item << " (all interfaces)\n"; continue; }
            auto s = std::make_unique<httplib::Server>();
            auto h = [this](const httplib::Request &q, httplib::Response &r) { handle(q, r); };
            s->Get(".*", h); s->Post(".*", h); s->Put(".*", h); s->Patch(".*", h); s->Delete(".*", h); s->Options(".*", h);
            httplib::Server *raw = s.get(); servers.push_back(std::move(s));
            std::thread([raw, host, port] {      // retry: the tailnet address may come up after omnid at boot
                for (int tries = 0; !g_stop; tries++) {
                    if (raw->listen(host, port)) break;
                    if (tries % 12 == 0) std::cerr << "[omnid] web listen failed on " << host << ":" << port << " (retrying)\n";
                    std::this_thread::sleep_for(std::chrono::seconds(5));
                }
            }).detach();
            std::cout << "[omnid] web face on " << host << ":" << port << std::endl;
        }
    }
    void stop() { for (auto &s : servers) s->stop(); }
};


// ------------------------------------------------------------------------------------------------ worker forwarding
// Handlers that still live in their own process are mounted behind omnid's front door: exec -> fleet-execd (Python
// fork-server), heads -> ring_headd (C++). One door, one protocol; the workers fold in-process in later phases.
static json forward_unix(const std::string &path, const json &payload, bool newline_terminated) {
    int s = socket(AF_UNIX, SOCK_STREAM, 0);
    sockaddr_un a{}; a.sun_family = AF_UNIX; strncpy(a.sun_path, path.c_str(), sizeof a.sun_path - 1);
    if (connect(s, (sockaddr *)&a, sizeof a) < 0) { close(s); throw std::runtime_error("worker unreachable: " + path); }
    std::string msg = payload.dump() + (newline_terminated ? "\n" : "");
    for (size_t off = 0; off < msg.size();) { ssize_t w = send(s, msg.data() + off, msg.size() - off, MSG_NOSIGNAL); if (w <= 0) break; off += w; }
    shutdown(s, SHUT_WR);
    std::string buf; char tmp[65536]; ssize_t n;
    while ((n = recv(s, tmp, sizeof tmp, 0)) > 0) buf.append(tmp, n);
    close(s);
    return json::parse(buf);
}
static std::string env_or(const char *k, const char *d) { return getenv(k) ? getenv(k) : d; }
static std::string env_or(const char *k, const std::string &d) { return getenv(k) ? getenv(k) : d; }


// ------------------------------------------------------------------------------------------------ MCP face
// `omnid mcp`: a thin stdio MCP server (JSON-RPC 2.0, newline-delimited) for opencode / Claude Code / any MCP client. It
// loads no model; every tool call is forwarded to the resident omnid on OMNID_SOCK. Tools: fleet_exec, embed, ring_heads, shard_search, shard_put, session_recall.
static json mcp_tools() {
    return json::array({
        {{"name", "fleet_exec"}, {"description", "Run bin/fleet-exec (the omnitool): a batch of bash/read/write/patch/sql/glob/grep "
            "operations, validated whole before anything runs. Input: {operations:[{type:'bash',command:'...'}, ...]}."},
         {"inputSchema", {{"type", "object"}, {"properties", {{"operations", {{"type", "array"}, {"items", {{"type", "object"}}}}},
                                                                {"cwd", {{"type", "string"}}}, {"concurrency", {{"type", "string"}, {"enum", {"auto", "parallel", "sequential"}}}}}}, {"required", {"operations"}}}}},
        {{"name", "embed"}, {"description", "Local text embeddings (no cloud): nomic-embed-text v1.5 (768-d) or BitNet-embedding-270M (640-d)."},
         {"inputSchema", {{"type", "object"}, {"properties", {{"texts", {{"type", "array"}, {"items", {{"type", "string"}}}}},
             {"model", {{"type", "string"}, {"enum", {"nomic", "bitnet270m"}}}}, {"kind", {{"type", "string"}, {"enum", {"document", "query", "raw"}}}}}},
             {"required", {"texts"}}}}},
        {{"name", "ring_heads"}, {"description", "Query the fleet's ring heads (ring_headd): {health:true} | {heads:true} | {head, text|texts}."},
         {"inputSchema", {{"type", "object"}}}},
        {{"name", "shard_search"}, {"description", "Search the fleet's memory (Claude Code sessions, the vault, stored notes): full-text + local "
            "nomic vectors, fused; every hit names its store, id and source. Input: {q, stores?:['sessions','vault','notes'], k?}."},
         {"inputSchema", {{"type", "object"}, {"properties", {{"q", {{"type", "string"}}}, {"stores", {{"type", "array"}, {"items", {{"type", "string"}}}}},
             {"k", {{"type", "integer"}}}}}, {"required", {"q"}}}}},
        {{"name", "arc"}, {"description", "Optional reasoning worker: exact engines (every reasoning generator), independent judges, pages and "
            "receipts. Input: {op: solve|verify|catalog|page_get|page_put|address|decode|ping, ...}: solve {domain, problem? | i, seed, level} → exact "
            "trace + answer + receipt; verify {domain, problem, answer} → the judge's verdict + receipt; page_put {domain, key, text, verified:true}."},
         {"inputSchema", {{"type", "object"}, {"properties", {{"op", {{"type", "string"}}}, {"domain", {{"type", "string"}}}}}, {"required", {"op"}}}}},
        {{"name", "shard_put"}, {"description", "Store a note in the shared memory any harness can search later. Input: {text, tags?, source?}."},
         {"inputSchema", {{"type", "object"}, {"properties", {{"text", {{"type", "string"}}}, {"tags", {{"type", "array"}, {"items", {{"type", "string"}}}}},
             {"source", {{"type", "string"}}}}}, {"required", {"text"}}}}},
        {{"name", "mark"}, {"description", "Record an app happening in the local event store of the optional worker (kind=event): {target:'<app>/<area>', outcome?, weight?, detail?}. "
            "The omniring area field reads it; evidence only, never a trigger."},
         {"inputSchema", {{"type", "object"}, {"properties", {{"target", {{"type", "string"}}}, {"outcome", {{"type", "string"}}},
             {"weight", {{"type", "number"}}}, {"detail", {{"type", "object"}}}}}, {"required", {"target"}}}}},
        {{"name", "session_recall"}, {"description", "Read back a past session's messages: {session} or {q} (best-matching session), limit?."},
         {"inputSchema", {{"type", "object"}, {"properties", {{"session", {{"type", "string"}}}, {"q", {{"type", "string"}}}, {"limit", {{"type", "integer"}}}}}}}},
    });
}
static bool unix_alive(const std::string &path);
// Tools backed by an optional worker (shardd / arcd) are listed only while that worker's socket answers, so a build
// without the workers advertises exactly what it can do.
static json mcp_tools_available() {
    const bool shard = unix_alive(env_or("SHARDD_SOCK", "/tmp/omnid-shard.sock"));
    const bool arc = unix_alive(env_or("ARCD_SOCK", "/tmp/omnid-arc.sock"));
    json out = json::array();
    for (auto &tl : mcp_tools()) {
        const std::string n = tl.value("name", std::string());
        if ((n == "shard_search" || n == "shard_put" || n == "session_recall" || n == "mark") && !shard) continue;
        if (n == "arc" && !arc) continue;
        out.push_back(tl);
    }
    return out;
}

static int mcp_main() {
    std::string sock = env_or("OMNID_SOCK", "/tmp/omnid.sock"), line;
    auto reply = [](const json &id, const json &result) { std::cout << json{{"jsonrpc", "2.0"}, {"id", id}, {"result", result}}.dump() << "\n" << std::flush; };
    auto fail = [](const json &id, int code, const std::string &m) { std::cout << json{{"jsonrpc", "2.0"}, {"id", id}, {"error", {{"code", code}, {"message", m}}}}.dump() << "\n" << std::flush; };
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        json m; try { m = json::parse(line); } catch (...) { fail(nullptr, -32700, "parse error"); continue; }
        if (!m.contains("id")) continue;                                   // notifications (initialized, cancelled)
        json id = m["id"]; std::string method = m.value("method", "");
        try {
            if (method == "initialize")
                reply(id, {{"protocolVersion", m["params"].value("protocolVersion", "2024-11-05")},
                           {"capabilities", {{"tools", json::object()}}}, {"serverInfo", {{"name", "omnid"}, {"version", "0.1"}}}});
            else if (method == "ping") reply(id, json::object());
            else if (method == "tools/list") reply(id, {{"tools", mcp_tools_available()}});
            else if (method == "tools/call") {
                std::string name = m["params"].value("name", ""); json a = m["params"].value("arguments", json::object());
                json r; std::string text; bool err = false;
                // Hoisted so the std::string outlives the json built from it (OMNID_DEFAULT_CWD).
                std::string default_cwd = env_or("OMNID_DEFAULT_CWD", std::string(home()));
                if (name == "fleet_exec") {
                    json payload = {{"operations", a.value("operations", json::array())}, {"concurrency", a.value("concurrency", "auto")}};
                    r = forward_unix(sock, {{"op", "exec"}, {"argv", {payload.dump()}}, {"cwd", a.value("cwd", default_cwd)},
                                            {"env", json::object()}, {"stdin", ""}}, true);
                    if (!r.value("ok", false)) { err = true; text = r.value("error", "omnid error"); }
                    else { text = r.value("stdout", ""); if (!r.value("stderr", "").empty()) text += "\n[stderr]\n" + r.value("stderr", ""); err = r.value("code", 1) != 0; }
                } else if (name == "embed") {
                    r = forward_unix(sock, {{"op", "embed"}, {"texts", a.value("texts", json::array())}, {"model", a.value("model", "nomic")},
                                            {"kind", a.value("kind", "document")}}, true);
                    err = !r.value("ok", false); text = err ? r.value("error", "") : json{{"dim", r["dim"]}, {"vectors", r["vectors"]}}.dump();
                } else if (name == "ring_heads") {
                    json q = a; q["op"] = "heads"; r = forward_unix(sock, q, true);
                    err = !r.value("ok", false); text = err ? r.value("error", "") : r["result"].dump();
                } else if (name == "arc") {                              // -> resident omnid -> arcd
                    json q = a; q["op"] = "arc." + a.value("op", std::string("ping")); r = forward_unix(sock, q, true);
                    err = !r.value("ok", false); text = r.dump();
                } else if (name == "shard_search" || name == "shard_put" || name == "session_recall" || name == "mark") {
                    json q = a; q["op"] = name; r = forward_unix(sock, q, true);
                    err = !r.value("ok", false); text = err ? r.value("error", "") : r.dump();
                } else { fail(id, -32602, "unknown tool: " + name); continue; }
                reply(id, {{"content", {{{"type", "text"}, {"text", text}}}}, {"isError", err}});
            } else fail(id, -32601, "method not found: " + method);
        } catch (const std::exception &ex) {
            reply(id, {{"content", {{{"type", "text"}, {"text", std::string("omnid: ") + ex.what()}}}}, {"isError", true}});
        }
    }
    return 0;
}

// ------------------------------------------------------------------------------------------------ dispatch + face
struct Omnid {
    EmbedHandler embed;
    std::atomic<long> reqs{0}, errs{0};
    explicit Omnid(int threads) : embed(threads) {}
    json dispatch(const json &req) {
        std::string op = req.value("op", "");
        if (op == "ping") {
            json models = json::array(); for (auto &kv : embed.specs) models.push_back(kv.first);
            return {{"ok", true}, {"server", "omnid"}, {"version", "0.1"}, {"ops", {"ping", "embed", "exec", "heads", "shard_search", "shard_put", "session_recall", "mark", "arc.*"}},
                    {"models", models}, {"requests", reqs.load()}, {"errors", errs.load()}};
        }
        if (op == "embed") return embed.call(req);
        if (op == "exec") {          // {op:exec, argv:[fleet-exec args, no program name], cwd, env, stdin} -> fleet-execd -> {stdout, stderr, code}
            json p = req; p.erase("op");
            json r = forward_unix(env_or("FLEET_EXEC_SOCK", "/tmp/fleet-execd.sock"), p, true);
            r["ok"] = true; return r;
        }
        if (op == "heads") {         // {op:heads, ...ring_headd request} -> ring_headd
            json p = req; p.erase("op");
            json r = forward_unix(env_or("RING_HEADD_SOCK", "/tmp/ring_headd.sock"), p, false);
            if (r.contains("error")) throw std::runtime_error("ring_headd: " + r["error"].get<std::string>());
            return {{"ok", true}, {"result", r}};
        }
        if (op == "shard_search" || op == "shard_put" || op == "session_recall" || op == "mark")   // optional worker ops -> shardd.py (warm Python worker)
            return forward_unix(env_or("SHARDD_SOCK", "/tmp/omnid-shard.sock"), req, true);
        if (op.rfind("arc.", 0) == 0) {                                  // optional reasoning worker -> arcd.py (exact engines, judges, pages, receipts)
            json q = req; q["op"] = op.substr(4);
            return forward_unix(env_or("ARCD_SOCK", "/tmp/omnid-arc.sock"), q, true);
        }
        throw std::runtime_error("unknown op: " + op);
    }
    void serve_conn(int c) {
        std::string buf; char tmp[65536];
        for (;;) {
            ssize_t n = recv(c, tmp, sizeof tmp, 0);
            if (n <= 0) break;
            buf.append(tmp, n);
            size_t nl;
            while ((nl = buf.find('\n')) != std::string::npos) {
                std::string line = buf.substr(0, nl); buf.erase(0, nl + 1);
                if (line.empty()) continue;
                json resp;
                try { resp = dispatch(json::parse(line)); reqs++; }
                catch (const std::exception &ex) { errs++; resp = {{"ok", false}, {"error", ex.what()}}; }
                std::string s = resp.dump() + "\n";
                size_t off = 0;
                while (off < s.size()) { ssize_t w = send(c, s.data() + off, s.size() - off, MSG_NOSIGNAL); if (w <= 0) break; off += w; }
            }
        }
        close(c);
    }
};

// ------------------------------------------------------------------------------------------------ shardd supervision
// shardd (optional shared-memory and event worker) is a warm Python worker. omnid keeps it alive as a child so it
// starts, restarts and stops with omnid.service (no separate unit). If its socket already answers (e.g. started by
// hand), omnid leaves it alone. OMNID_SHARDD=0 disables; OMNID_SHARDD_CMD overrides the command.
static bool unix_alive(const std::string &path) {
    int fd = socket(AF_UNIX, SOCK_STREAM, 0); if (fd < 0) return false;
    sockaddr_un a{}; a.sun_family = AF_UNIX; strncpy(a.sun_path, path.c_str(), sizeof a.sun_path - 1);
    bool ok = connect(fd, (sockaddr *)&a, sizeof a) == 0; close(fd); return ok;
}
static void supervise_shardd(std::atomic<bool> &stop) {
    if (getenv("OMNID_SHARDD") && std::string(getenv("OMNID_SHARDD")) == "0") return;
    std::string sock = env_or("SHARDD_SOCK", "/tmp/omnid-shard.sock");
    // Resolve shardd.py next to this binary, so a checkout anywhere still
    // works. OMNID_SHARDD_CMD stays the full override.
    std::string self = env_or("OMNID_HOME", "");
    if (self.empty()) self = home() + "/omnid";
    if (!getenv("OMNID_SHARDD_CMD") && access((self + "/shardd.py").c_str(), R_OK) != 0) return;   // no worker shipped: nothing to supervise
    std::string cmd = env_or("OMNID_SHARDD_CMD", ("python3 " + self + "/shardd.py").c_str());
    std::string logp = env_or("OMNID_SHARDD_LOG", "/tmp/omnid-shardd.log");
    pid_t child = -1; int backoff = 2;
    while (!stop) {
        if (child > 0 && waitpid(child, nullptr, WNOHANG) == child) { std::cerr << "[omnid] shardd exited; restarting\n"; child = -1; }
        if (child < 0 && !unix_alive(sock)) {
            child = fork();
            if (child == 0) {
                setsid();
                int lf = open(logp.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0600);
                if (lf >= 0) { dup2(lf, 1); dup2(lf, 2); }
                execl("/bin/sh", "sh", "-c", ("exec " + cmd).c_str(), (char *)nullptr);
                _exit(127);
            }
            std::cerr << "[omnid] started shardd pid " << child << "\n";
            std::this_thread::sleep_for(std::chrono::seconds(backoff)); backoff = std::min(backoff * 2, 60);
        } else if (unix_alive(sock)) backoff = 2;
        std::this_thread::sleep_for(std::chrono::seconds(5));
    }
    if (child > 0) { kill(child, SIGTERM); waitpid(child, nullptr, 0); }
}

int main(int argc, char **argv) {
    std::string sock = getenv("OMNID_SOCK") ? getenv("OMNID_SOCK") : "/tmp/omnid.sock";
    if (argc >= 2 && std::string(argv[1]) == "mcp") return mcp_main();
    int threads = getenv("OMNID_THREADS") ? std::atoi(getenv("OMNID_THREADS")) : 4;
    llama_backend_init();
    llama_log_set([](ggml_log_level l, const char *t, void *) { if (l >= GGML_LOG_LEVEL_ERROR) fputs(t, stderr); }, nullptr);
    Omnid srv(threads);
    if (argc >= 2 && std::string(argv[1]) == "--once") {        // one request from stdin (tests / scripts)
        std::string line; std::getline(std::cin, line);
        try { std::cout << srv.dispatch(json::parse(line)).dump() << std::endl; return 0; }
        catch (const std::exception &ex) { std::cout << json{{"ok", false}, {"error", ex.what()}}.dump() << std::endl; return 1; }
    }
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    sockaddr_un a{}; a.sun_family = AF_UNIX; strncpy(a.sun_path, sock.c_str(), sizeof a.sun_path - 1);
    unlink(sock.c_str());
    if (bind(fd, (sockaddr *)&a, sizeof a) < 0 || listen(fd, 128) < 0) { perror("omnid bind/listen"); return 1; }
    chmod(sock.c_str(), 0600);
    signal(SIGINT, on_signal); signal(SIGTERM, on_signal); signal(SIGPIPE, SIG_IGN);
    std::atomic<bool> stop_sup{false};
    std::thread sup([&stop_sup] { supervise_shardd(stop_sup); });
    WebFace web;
    if (getenv("OMNID_WEB")) { try { web.start(getenv("OMNID_WEB")); } catch (const std::exception &ex) { std::cerr << "[omnid] web: " << ex.what() << "\n"; } }
    std::cout << "[omnid] listening on " << sock << " threads=" << threads << std::endl;
    while (!g_stop) {
        timeval tv{1, 0}; fd_set rs; FD_ZERO(&rs); FD_SET(fd, &rs);
        if (select(fd + 1, &rs, nullptr, nullptr, &tv) <= 0) continue;
        int c = accept(fd, nullptr, nullptr);
        if (c < 0) continue;
        std::thread([&srv, c] { srv.serve_conn(c); }).detach();
    }
    stop_sup = true; if (sup.joinable()) sup.join();
    web.stop(); close(fd); unlink(sock.c_str());
    std::cout << "[omnid] stopped: requests=" << srv.reqs << " errors=" << srv.errs << std::endl;
    llama_backend_free();
}
