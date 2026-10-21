# ollama-council

**Small local models make mistakes. Two of them rarely make the *same* mistake.**

`ollama-council` asks two different model families the same question, checks
whether they actually agree (by meaning, not word overlap), and only calls in a
third model when they don't. You get answers cross-checked by more than one
model at close to single-model cost, fully offline, with zero dependencies.

```console
$ council "In one sentence: what is the capital of Japan?"
The capital of Japan is Tokyo.
[agreed · agreement 1.00 · 14.0s]

$ council --json 'Return JSON {"sentiment": "positive|negative"} for: I love this song'
{"sentiment": "positive"}
[agreed · agreement 1.00 · 3.8s]
```

## How it works

```
        question
           │
   ┌───────┴────────┐
   ▼                ▼
 gemma2:2b       llama3.2          two model families answer independently
   │                │
   └──── agree? ────┘              prose: embedding cosine ≥ 0.80 (nomic-embed-text)
        │      │                   JSON:  share of matching short fields ≥ 0.5
       yes     no / a model failed
        │      ▼
        │   qwen2.5:3b joins
        │      ├─ JSON  → majority vote (a real model's JSON, never a merge)
        │      └─ prose → synthesizer merges the answers
        ▼
   verify → redact secrets → cache 6h → answer
```

**Why "by meaning":** word overlap scored two answers that both said
`"risk": "high"` as 0.18 agreement. Comparing embeddings for prose, and the
actual field values for JSON, fixes that.

On an Intel MacBook Pro (CPU only), a benchmark of 4 real questions took
**327 s / 9 model calls** in adaptive mode vs **708 s / 14 calls** asking all
three models every time, and every answer was valid.

## Built in

- **Streaming with hard deadlines.** Closing the stream cancels generation in
  Ollama. Non-streamed requests keep running after the client gives up and pile
  into a backlog.
- **Cross-process FIFO queue.** Ten scripts sharing one CPU-bound Ollama won't
  stampede it: they take turns in arrival order. A council holds one slot for its
  whole decision. Set `COUNCIL_SLOTS=2` for more parallelism.
- **Prompt-injection screen.** Suspicious text ("ignore previous
  instructions…") is flagged and wrapped as data, not blocked.
- **Answer checks.** Empty, refused or broken-JSON answers are replaced by a
  valid answer from another member.
- **Secret redaction.** Anthropic/OpenAI/GitHub/AWS/Google/Slack keys and
  private keys are stripped from answers.
- **Cache.** Identical questions within 6 hours are answered instantly (SQLite).
- **No dependencies.** Python stdlib only.

## Install

```bash
pip install git+https://github.com/naviidaviimusic/ollama-council
ollama pull gemma2:2b && ollama pull llama3.2 && ollama pull qwen2.5:3b
ollama pull nomic-embed-text      # optional, for meaning-based agreement
```

## Use it in Python

```python
from ollama_council import Council

council = Council()                                  # gemma2:2b, llama3.2, qwen2.5:3b
r = council.ask("Is this email a phishing attempt? ...", schema="json")
print(r.answer)        # the answer
print(r.mode)          # agreed | escalated-vote | escalated-synthesis | cache | ...
print(r.agreement)     # 0.0 – 1.0
print(r.members)       # each model's own answer

# your own lineup, every model every time
Council(members=["qwen2.5:7b", "llama3.1:8b", "gemma2:9b"], strategy="full").ask("...")
```

`schema` can be `"json"` or a JSON-schema dict; Ollama then constrains decoding to it.

## CLI

```
council [-m models] [-s system] [--json] [--full] [--no-cache] [--timeout s] [-v] question
```

`-v` prints each member's answer. The question can also come from stdin.

## Configuration

| Env var | Default | |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server |
| `COUNCIL_SLOTS` | `1` | concurrent model calls across all processes |
| `COUNCIL_QUEUE_DIR` | `~/.cache/ollama-council/queue` | queue location |

## Development

```bash
pip install -e ".[dev]" && pytest -q && ruff check .
```

Tests use fake models, so they run without Ollama.

## License

MIT © Victor Navejar ([Navii](https://github.com/naviidaviimusic))
