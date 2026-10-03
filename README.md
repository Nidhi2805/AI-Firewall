# AI Firewall

A document-grounded assistant with a security firewall on every layer. Upload a
document — an insurance policy, a certificate, a contract — and ask questions
about it. The assistant answers **only** from that document, cites where each
answer came from, refuses jailbreaks and prompt-injection attempts, and logs
every decision so it can be audited.

Built to be run by anyone: clone it, add an API key, and it works end to end.
Without a key it still runs (it returns the best retrieved passage instead of a
generated answer), so a demo never hard-fails.

---

## What it does

- **Grounded, cited answers.** Every answer is drawn only from the uploaded
  document and tagged with the passage it came from. No outside knowledge, no
  hallucinated facts.
- **A firewall on input and output.** Jailbreaks, prompt injection, and PII are
  caught before they reach the model, and the model's own output is scanned
  before it reaches the user. Agent tool-calls go through a separate action
  firewall (argument inspection + dangerous-sequence detection).
- **Honest abstention.** If the answer isn't in the document, it says so rather
  than guessing.
- **A visible, editable knowledge base.** See the exact text the assistant uses,
  edit any line in place and re-index — no need to re-upload to fix one line —
  and see how the document is split into passages.
- **An activity dashboard.** Every question, decision, block, and latency,
  rolled into live metrics.
- **Fast, multi-provider generation.** Groq first (low latency), Gemini as an
  automatic fallback, retrieved-passage as a last resort.

## The pipeline

```
question
  → conversational gate   (greetings get an instant reply)
  → input checkpoint      (benign / jailbreak / injection / PII — threats blocked)
  → topic gate            (is it answerable from this document? if not, abstain)
  → retriever + reranker  (most relevant passages)
  → relevance floor       (nothing good enough? abstain)
  → LLM generation        (Groq → Gemini → passage; grounded + cited)
  → output checkpoint     (scan the answer for leaked PII / injection)
  → audit log             (record the decision + latency)
```

---

## Quick start

Requires Python 3.9+.

```bash
# 1. clone and enter
git clone <your-repo-url> ai_firewall && cd ai_firewall

# 2. create a virtual environment
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 3. install
pip install -r requirements.txt

# 4. configure (copy the template, then edit .env with your key)
cp .env.example .env

# 5. generate data + train the classifiers (one time)
python scripts/setup.py

# 6. run
python app.py                        # open http://localhost:5000
```

Register an account, log in, upload a document on the **Knowledge base** page,
and start asking questions.

## Getting an API key

The assistant uses **Groq** by default (fast, free tier).

1. Create a key at <https://console.groq.com/keys> (starts with `gsk_`).
2. Put it in `.env` as `GROQ_API_KEY`.

**Model IDs change often.** If you see a "model not found" error, list the models
your key can actually use and set one of them:

```bash
python scripts/list_models.py
# then set GROQ_MODEL in .env to a chat model from that list
```

A Gemini key (`GEMINI_API_KEY`) is optional. It serves as an automatic fallback
and powers native-PDF answering for scanned documents whose extracted text is
poor — see *Known limitations*.

---

## Testing

```bash
python tests/test_pipeline.py        # or: python -m pytest tests/ -q
```

Nine regression tests cover threat blocking, abstention, the relevance floor,
the output checkpoint, and the action firewall.

## Deployment

The app is a standard Flask/WSGI app. Two things matter in production, and both
are already configured:

- **Run a single worker.** Uploaded knowledge bases and sessions are held in
  memory per process, so multiple workers would not share them. The included
  `Procfile` and `render.yaml` use `gunicorn -w 1`.
- **Set secrets in the host's environment**, never in the repo. `.env` is for
  local use only and is gitignored.

**Render:** point a new Blueprint service at this repo; it reads `render.yaml`.
Set `GROQ_API_KEY` (and optionally `GEMINI_API_KEY`) in the dashboard.

**Any host / locally public:** run `python app.py` and expose it with a tunnel:
```bash
cloudflared tunnel --url http://localhost:5000
```

---

## Project structure

```
ai_firewall/
├── app.py                  Flask app: auth, chat, KB editor, activity dashboard
├── requirements.txt
├── .env.example            copy to .env and add your keys
├── Procfile                production start command (single worker)
├── render.yaml             Render deploy blueprint
├── scripts/
│   ├── setup.py            one-time: generate data + train models
│   └── list_models.py      print the models your API key can use
├── src/
│   ├── pipeline.py         orchestrator (+ latency, audit, abstention)
│   ├── llm.py              multi-provider generation (Groq → Gemini → passage)
│   ├── audit_log.py        append-only log + dashboard summary
│   ├── knowledge_base.py   upload → chunk → index → fresh topic gate
│   ├── input_checkpoint.py classical-ML threat classifier
│   ├── topic_gate.py       in-domain / answerability gate
│   ├── retriever.py reranker.py vector_store.py chunker.py
│   ├── context_scanner.py  ingestion-time PII / injection tagging
│   ├── action_firewall.py  agent tool-call firewall
│   ├── conversational_gate.py sentiment.py features.py
│   └── document_extractor.py auth.py generate_dataset.py display.py
└── tests/test_pipeline.py
```

---

## Known limitations

These are real and worth stating plainly.

- **Keyword-based threat detection has a ceiling.** There is always another
  jailbreak phrasing not yet on the list. Reliability scales with adversarial
  testing and training-set size, not with how finished the code looks. A
  production hardening step would add an LLM-based classifier behind the fast
  classical one.
- **A single-document knowledge base weakens the topic gate.** With only one
  document it cannot learn a sharp in/out boundary, so a lexically-overlapping
  out-of-scope question can occasionally be mis-gated. Grounded generation
  ("answer only from context") is the backstop.
- **Scanned / heavily tabular PDFs extract poorly.** Text extraction flattens
  complex tables into noise, which caps retrieval quality on number-heavy
  questions. The native-PDF path (via Gemini) is the robust route for these;
  editing the text by hand on the Knowledge base page is another.
- **State is in-memory.** Uploaded KBs and sessions live in the process; a
  restart clears them, and the app must run as a single worker. Moving state to
  a database is the next step for real multi-user production.
- **The bundled dataset is synthetic.** It proves the pipeline's logic, not
  real-world accuracy. Swapping in a real labelled attack set is the natural
  next step.

## Security

- Never commit `.env` or any real API key. The repo's `.gitignore` excludes it.
- If a key is ever exposed, rotate it immediately at the provider's console.
- Passwords are hashed (werkzeug). The file-backed account store is fine for a
  prototype; use a real database before production.
