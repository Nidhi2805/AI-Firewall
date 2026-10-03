"""
The product — a login-gated, document-grounded chat with a live security
firewall, answer citations, custom knowledge-base upload, and an activity
dashboard showing what the firewall checked and blocked.

Routes:
  /login /register /logout      — accounts (auth.py)
  /                             — chat (grounded answers + firewall trace)
  /kb                           — see what the assistant knows; upload your own doc
  /admin                        — firewall dashboard (audit_log summary)
  /api/chat                     — JSON chat endpoint
"""

import os
from dotenv import load_dotenv
from flask import (Flask, request, session, redirect, url_for,
                   render_template_string, jsonify)

from src.auth import register_user, verify_login
from src.document_extractor import extract_text, ExtractionError
from src.knowledge_base import build_custom_kb, default_kb_description
from src.pipeline import AIFirewallPipeline
from src import audit_log

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("APP_SECRET", "dev-secret-change-me")

# Per-user pipeline + KB state (in-memory; fine for a prototype/demo).
_pipelines = {}
_kb_desc = {}
_pdf_bytes = {}
_kb_text = {}          # user -> {"text": raw text the KB was built from, "source": name}


def _get_pipeline(user):
    if user not in _pipelines:
        _pipelines[user] = AIFirewallPipeline(user=user)
        _kb_desc[user] = default_kb_description()
    return _pipelines[user]


def _require_login():
    return session.get("user")


# ---- shared UI (design system) ----------------------------------------
HEAD = """
<!DOCTYPE html><html lang="en" data-theme="light"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>AI Firewall</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Inter+Tight:wght@500;600;700&display=swap" rel="stylesheet">
<style>
:root{
  --paper:#F4F2EC;--ink:#33373D;--muted:#8A8F98;--line:#E7E4DC;
  --steel:#4A6B96;--steel-soft:#EDF1F6;--green:#3C8A5C;--green-soft:#EAF2EC;
  --red:#C06A5E;--red-soft:#F7EBE8;--amber:#B98A4E;--amber-soft:#F6EEE1;--card:#FBFAF6;
  font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;box-sizing:border-box;
}
*,*::before,*::after{box-sizing:inherit}
body{margin:0;background:var(--paper);color:var(--ink);line-height:1.5;
  padding-top:env(safe-area-inset-top,0);padding-bottom:env(safe-area-inset-bottom,0)}
h1,h2,h3{font-family:'Inter Tight',sans-serif;margin:0;letter-spacing:-.01em}
h2{font-size:22px;margin:18px 0 4px}
a{color:var(--steel)}
.wrap{max-width:860px;margin:0 auto;padding:0 20px}
header{border-bottom:1px solid var(--line);background:rgba(244,242,236,.85);backdrop-filter:blur(8px);position:sticky;top:0;z-index:10}
.bar{display:flex;align-items:center;justify-content:space-between;height:60px}
.brand{display:flex;align-items:center;gap:10px;font-family:'Inter Tight';font-weight:700;font-size:17px}
.seal{width:26px;height:26px;border-radius:6px;background:var(--steel);position:relative;flex:none}
.seal::after{content:"";position:absolute;inset:7px;border:2px solid #fff;border-radius:2px}
nav{display:flex;gap:4px;flex-wrap:wrap}
nav a{font-size:14px;color:var(--muted);text-decoration:none;padding:7px 12px;border-radius:7px}
nav a:hover{background:#EEEDE8;color:var(--ink)}
nav a.on{color:var(--steel);background:var(--steel-soft);font-weight:500}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px;margin:16px 0}
.docstrip{display:flex;align-items:center;gap:10px;padding:11px 14px;margin:18px 0 6px;background:var(--card);border:1px solid var(--line);border-radius:10px;font-size:13.5px}
.docstrip .dot{width:7px;height:7px;border-radius:50%;background:var(--green);flex:none}
.log{padding:14px 0 130px}
.turn{margin:22px 0}
.q{font-family:'Inter Tight';font-weight:600;font-size:16.5px;margin-bottom:10px}
.q::before{content:"";display:inline-block;width:3px;height:15px;background:var(--steel);margin-right:9px;vertical-align:-2px;border-radius:2px}
.trace{display:flex;gap:6px;flex-wrap:wrap;margin:0 0 10px 12px;font-size:12px}
.step{display:inline-flex;align-items:center;gap:6px;padding:3px 10px;border-radius:20px;background:#F0EFEA;color:var(--muted);font-weight:500}
.step::before{content:"";width:6px;height:6px;border-radius:50%;background:#C9C7BF}
.step.pass{background:var(--green-soft);color:var(--green)}
.step.pass::before{background:var(--green)}
.step.block{background:var(--red-soft);color:var(--red)}
.step.block::before{background:var(--red)}
.answer{margin-left:12px;font-size:15.5px;max-width:65ch}
.cites{margin-top:10px;display:flex;gap:6px;flex-wrap:wrap}
.cite{font-size:12px;color:var(--steel);background:var(--steel-soft);padding:3px 9px;border-radius:6px;font-variant-numeric:tabular-nums}
.note-red{margin-left:12px;font-size:14.5px;color:var(--red);background:var(--red-soft);border:1px solid #F3C9C4;padding:11px 13px;border-radius:9px;max-width:60ch}
.note-grey{margin-left:12px;font-size:14.5px;color:var(--muted);background:#F0EFEA;padding:11px 13px;border-radius:9px;max-width:55ch}
input,textarea{font:inherit;color:var(--ink);background:var(--card);border:1.5px solid var(--line);border-radius:10px;padding:11px 13px;width:100%}
input:focus,textarea:focus{outline:0;border-color:var(--steel)}
textarea{resize:vertical}
button{font:inherit;font-weight:600;font-size:14px;border:0;background:var(--steel);color:#fff;padding:11px 18px;border-radius:9px;cursor:pointer}
button:hover{background:#3E5C82}
.composer{position:fixed;bottom:0;left:0;right:0;background:linear-gradient(transparent,var(--paper) 22%);padding:20px 20px calc(18px + env(safe-area-inset-bottom,0))}
.composer .inner{max-width:860px;margin:0 auto;display:flex;gap:10px}
.metrics{display:flex;gap:12px;flex-wrap:wrap;margin-top:8px}
.metric{flex:1;min-width:120px;background:var(--card);border:1px solid var(--line);border-radius:11px;padding:15px}
.metric b{display:block;font-family:'Inter Tight';font-size:28px;color:var(--steel);font-variant-numeric:tabular-nums}
.metric span{font-size:13px;color:var(--muted)}
.row{border-top:1px solid var(--line);padding:9px 2px;font-size:14px;display:flex;align-items:center;gap:9px}
.row:first-child{border-top:0}
.pill{padding:2px 9px;border-radius:20px;font-size:12px;font-weight:500;flex:none}
.pill.ok{background:var(--green-soft);color:var(--green)}
.pill.block{background:var(--red-soft);color:var(--red)}
.pill.ood{background:var(--amber-soft);color:var(--amber)}
.rowmeta{margin-left:auto;color:var(--muted);font-size:12px;font-variant-numeric:tabular-nums}
.passage{border-top:1px solid var(--line);padding:10px 0;font-size:13.5px;color:var(--muted)}
.passage:first-of-type{border-top:0}
.auth{max-width:380px;margin:60px auto}
.err{color:var(--red);font-size:14px;margin-top:10px}
.ok-msg{color:var(--green);font-size:14px;margin-top:10px}
.muted{color:var(--muted)}
small{color:var(--muted)}
label{font-size:13px;color:var(--muted);display:block;margin:12px 0 5px}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --paper:#1B1E22;--ink:#D2D5D9;--muted:#8C929A;--line:#2E333A;--card:#22262B;
  --steel:#89A8CE;--steel-soft:#273343;--green:#7FC79A;--green-soft:#1E2A22;
  --red:#D79389;--red-soft:#302220;--amber:#D7B27E;--amber-soft:#2C2619;}}
</style></head><body>
"""

FOOT = "</body></html>"


def nav(active):
    items = [("/", "Chat", "chat"), ("/kb", "Knowledge base", "kb"), ("/admin", "Activity", "admin")]
    links = "".join(
        '<a href="%s" class="%s">%s</a>' % (href, ("on" if key == active else ""), label)
        for href, label, key in items)
    return ('<header><div class="wrap bar"><div class="brand"><span class="seal"></span>'
            'AI Firewall</div><nav>' + links + '<a href="/logout">Sign out</a></nav></div></header>')


def _trace_for(result):
    """Firewall-trace chips + display kind from a pipeline result."""
    status = result.get("status")
    if status == "blocked_at_input":
        return [{"label": "Blocked at input", "cls": "block"}], "blocked"
    if status == "blocked_at_output":
        return [{"label": "Input check", "cls": "pass"}, {"label": "Blocked at output", "cls": "block"}], "blocked"
    if status == "short_circuited_out_of_domain":
        return [{"label": "Input check", "cls": "pass"}, {"label": "Out of scope", "cls": "block"}], "abstain"
    if status == "smalltalk_instant_reply":
        return [], "answer"
    if result.get("generation_source") == "no_relevant_context" or (result.get("response", "").startswith("I don't have")):
        return [{"label": "Input check", "cls": "pass"}, {"label": "In scope", "cls": "pass"},
                {"label": "No match in document", "cls": "block"}], "abstain"
    return [{"label": "Input check", "cls": "pass"}, {"label": "In scope", "cls": "pass"},
            {"label": "Retrieved", "cls": "pass"}, {"label": "Answered", "cls": "pass"}], "answer"


@app.route("/register", methods=["GET", "POST"])
def register():
    msg = ""
    if request.method == "POST":
        ok, msg = register_user(request.form.get("username", ""), request.form.get("password", ""))
        if ok:
            return redirect(url_for("login"))
    return render_template_string(HEAD + """
    <div class="wrap auth">
      <div class="brand" style="margin-bottom:20px"><span class="seal"></span>AI Firewall</div>
      <h2>Create your account</h2>
      <div class="card"><form method="post">
        <label>Username</label><input name="username" autofocus>
        <label>Password</label><input name="password" type="password" placeholder="At least 6 characters">
        <div style="margin-top:16px"><button>Create account</button></div>
      </form>{% if msg %}<div class="err">{{msg}}</div>{% endif %}</div>
      <small>Already have an account? <a href="/login">Sign in</a></small>
    </div>""" + FOOT, msg=msg)


@app.route("/login", methods=["GET", "POST"])
def login():
    msg = ""
    if request.method == "POST":
        u, p = request.form.get("username", ""), request.form.get("password", "")
        if verify_login(u, p):
            session["user"] = u.strip()
            session.setdefault("history_" + u.strip(), [])
            return redirect(url_for("chat"))
        msg = "That username and password don't match. Try again."
    return render_template_string(HEAD + """
    <div class="wrap auth">
      <div class="brand" style="margin-bottom:20px"><span class="seal"></span>AI Firewall</div>
      <h2>Sign in</h2>
      <div class="card"><form method="post">
        <label>Username</label><input name="username" autofocus>
        <label>Password</label><input name="password" type="password">
        <div style="margin-top:16px"><button>Sign in</button></div>
      </form>{% if msg %}<div class="err">{{msg}}</div>{% endif %}</div>
      <small>New here? <a href="/register">Create an account</a></small>
    </div>""" + FOOT, msg=msg)


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


@app.route("/")
def chat():
    user = _require_login()
    if not user:
        return redirect(url_for("login"))
    _get_pipeline(user)
    history = session.get("history_" + user, [])
    return render_template_string(HEAD + nav("chat") + """
    <div class="wrap">
      <div class="docstrip"><span class="dot"></span><span>{{desc}}</span></div>
      <div class="log">
      {% for turn in history %}
        <div class="turn">
          <div class="q">{{turn.q}}</div>
          {% if turn.trace %}<div class="trace">
            {% for s in turn.trace %}<span class="step {{s.cls}}">{{s.label}}</span>{% endfor %}
          </div>{% endif %}
          {% if turn.kind == 'blocked' %}<div class="note-red">{{turn.a}}</div>
          {% elif turn.kind == 'abstain' %}<div class="note-grey">{{turn.a}}</div>
          {% else %}<div class="answer">{{turn.a}}
            {% if turn.cites %}<div class="cites">
              {% for c in turn.cites %}<span class="cite" title="{{c.text}}">Passage {{c.marker}}</span>{% endfor %}
            </div>{% endif %}</div>{% endif %}
        </div>
      {% endfor %}
      {% if not history %}<div class="turn"><div class="note-grey" style="margin-left:0">
        Ask a question about the loaded document. Every question passes through the firewall first —
        you'll see each checkpoint it clears, or where it's stopped.</div></div>{% endif %}
      </div>
    </div>
    <div class="composer"><div class="inner">
      <form method="post" action="/ask" style="display:flex;gap:10px;width:100%">
        <input name="q" placeholder="Ask about the document…" autofocus><button>Ask</button>
      </form>
    </div></div>""" + FOOT, desc=_kb_desc.get(user, default_kb_description()), history=history)


@app.route("/ask", methods=["POST"])
def ask():
    user = _require_login()
    if not user:
        return redirect(url_for("login"))
    q = request.form.get("q", "").strip()
    if q:
        result = _get_pipeline(user).process_query(q)
        trace, kind = _trace_for(result)
        hist = session.get("history_" + user, [])
        hist.append({"q": q, "a": result["response"], "cites": result.get("citations", []),
                     "trace": trace, "kind": kind})
        session["history_" + user] = hist[-20:]
    return redirect(url_for("chat"))


@app.route("/api/chat", methods=["POST"])
def api_chat():
    user = _require_login()
    if not user:
        return jsonify({"error": "not authenticated"}), 401
    q = (request.json or {}).get("query", "").strip()
    result = _get_pipeline(user).process_query(q)
    return jsonify({"answer": result["response"], "status": result["status"],
                    "citations": result.get("citations", [])})


@app.route("/kb", methods=["GET"])
def kb():
    """View the knowledge base: the editable document text + indexed passages."""
    user = _require_login()
    if not user:
        return redirect(url_for("login"))
    _get_pipeline(user)
    kb_state = _kb_text.get(user, {"text": "", "source": None})
    pipe = _pipelines.get(user)
    chunks = []
    if pipe is not None:
        chunks = [{"doc_id": r["doc_id"], "text": r["text"], "tag": r.get("tag")}
                  for r in pipe.store.records]
    return render_template_string(HEAD + nav("kb") + """
    <div class="wrap">
      <h2>Knowledge base</h2>
      <div class="docstrip"><span class="dot"></span><span>{{desc}}</span></div>

      <div class="card">
        <h3 style="font-size:16px">Document text</h3>
        <small>This is exactly what the assistant answers from. Edit any line and save —
        you don't need to re-upload the whole document to fix one line.</small>
        <form method="post" action="/kb/save" style="margin-top:12px">
          <textarea name="text" style="min-height:300px;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:13px">{{kb_text}}</textarea>
          <div style="margin-top:12px"><button>Save &amp; re-index</button>
          {% if msg %}<span class="ok-msg" style="margin-left:10px">{{msg}}</span>{% endif %}</div>
        </form>
      </div>

      <div class="card">
        <h3 style="font-size:16px">Indexed passages <span class="muted">({{chunks|length}})</span></h3>
        <small>How the document is split for retrieval. Excluded passages (possible PII or injected
        text) are kept out of answers.</small>
        <div style="margin-top:10px">
        {% for c in chunks %}
          <div class="passage">
            <span class="pill {{ 'block' if c.tag=='flagged' else 'ok' }}">{{ 'excluded' if c.tag=='flagged' else 'indexed' }}</span>
            &nbsp;{{c.text}}
          </div>
        {% endfor %}
        {% if not chunks %}<div class="passage" style="border:0">Nothing indexed yet — add a document below.</div>{% endif %}
        </div>
      </div>

      <div class="card">
        <h3 style="font-size:16px">Replace the document</h3>
        <small>Upload a PDF, Word, or text file, or paste new text. This starts a fresh knowledge base
        and resets the chat.</small>
        <form method="post" action="/kb/upload" enctype="multipart/form-data" style="margin-top:12px">
          <input type="file" name="file" accept=".pdf,.docx,.txt,.md">
          <label>or paste text</label>
          <textarea name="text" rows="3" placeholder="Paste document text here"></textarea>
          <div style="margin-top:12px"><button>Rebuild knowledge base</button></div>
        </form>
      </div>
    </div>""" + FOOT, desc=_kb_desc.get(user, default_kb_description()),
    kb_text=kb_state["text"], chunks=chunks, msg=request.args.get("msg", ""))


@app.route("/kb/upload", methods=["POST"])
def kb_upload():
    """Replace the whole KB with a new uploaded/pasted document."""
    user = _require_login()
    if not user:
        return redirect(url_for("login"))
    msg = ""
    try:
        f = request.files.get("file")
        pasted = request.form.get("text", "").strip()
        if f and f.filename:
            data = f.read()
            text = extract_text(f.filename, data)
            _pdf_bytes[user] = data if f.filename.lower().endswith(".pdf") else None
            src_name = f.filename
        elif pasted:
            text = pasted
            _pdf_bytes[user] = None
            src_name = "pasted text"
        else:
            raise ExtractionError("Add a file or paste some text to build a knowledge base.")
        store, gate, desc = build_custom_kb(text, source_name=src_name)
        _pipelines[user] = AIFirewallPipeline(store=store, topic_gate=gate,
                                              pdf_bytes=_pdf_bytes[user], user=user)
        _kb_desc[user] = desc
        _kb_text[user] = {"text": text, "source": src_name}
        session["history_" + user] = []
        msg = "Knowledge base rebuilt. The chat has been reset to the new document."
    except ExtractionError as e:
        msg = str(e)
    return redirect(url_for("kb", msg=msg))


@app.route("/kb/save", methods=["POST"])
def kb_save():
    """Save an in-place edit to the document and re-index. No re-upload needed.

    Re-indexes the whole document on save — for a 50-page document this is still
    fast (under a second) and keeps the index consistent with the edited text."""
    user = _require_login()
    if not user:
        return redirect(url_for("login"))
    new_text = request.form.get("text", "").strip()
    if not new_text:
        return redirect(url_for("kb", msg="Nothing to save — the document was empty."))
    source = (_kb_text.get(user) or {}).get("source", "edited document")
    _pdf_bytes[user] = None   # edited text no longer matches any original scanned PDF
    store, gate, desc = build_custom_kb(new_text, source_name="%s (edited)" % source)
    _pipelines[user] = AIFirewallPipeline(store=store, topic_gate=gate, pdf_bytes=None, user=user)
    _kb_desc[user] = desc
    _kb_text[user] = {"text": new_text, "source": source}
    session["history_" + user] = []
    return redirect(url_for("kb", msg="Saved and re-indexed. The edit is now live."))


@app.route("/admin")
def admin():
    user = _require_login()
    if not user:
        return redirect(url_for("login"))
    s = audit_log.summary()
    return render_template_string(HEAD + nav("admin") + """
    <div class="wrap">
      <h2>Activity</h2><small>What the firewall has processed and blocked.</small>
      <div class="metrics">
        <div class="metric"><b>{{s.total_queries}}</b><span>questions</span></div>
        <div class="metric"><b>{{s.answered}}</b><span>answered</span></div>
        <div class="metric"><b>{{s.threats_blocked}}</b><span>threats blocked</span></div>
        <div class="metric"><b>{{s.out_of_domain}}</b><span>out of scope</span></div>
        <div class="metric"><b>{{s.avg_latency_ms}}</b><span>avg ms</span></div>
      </div>
      <div class="card"><h3 style="font-size:16px">Recent activity</h3>
      <div style="margin-top:6px">
      {% for e in s.recent %}
        <div class="row">
          {% if e.status in ['blocked_at_input','blocked_at_output'] %}<span class="pill block">blocked</span>
          {% elif e.status=='short_circuited_out_of_domain' %}<span class="pill ood">out of scope</span>
          {% else %}<span class="pill ok">answered</span>{% endif %}
          <span>{{e.query}}</span>
          <span class="rowmeta">{{e.latency_ms}} ms{% if e.input_label and e.input_label!='benign' %} · {{e.input_label}}{% endif %}</span>
        </div>
      {% endfor %}
      {% if not s.recent %}<div class="row" style="border:0">No activity yet. Ask a question in the chat.</div>{% endif %}
      </div></div>
    </div>""" + FOOT, s=s)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)