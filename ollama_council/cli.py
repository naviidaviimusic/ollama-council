"""council "question"  — ask the local model council from the shell."""

import argparse
import sys

from . import client
from .council import DEFAULT_MEMBERS, Council


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="council", description="Ask several local Ollama models and converge on one answer.")
    p.add_argument("question", nargs="*", help="question (or read from stdin)")
    p.add_argument("-m", "--members", default=",".join(DEFAULT_MEMBERS),
                   help="comma-separated Ollama models (default: %(default)s)")
    p.add_argument("-s", "--system", help="system prompt")
    p.add_argument("--json", action="store_true", help="constrain answers to JSON")
    p.add_argument("--full", action="store_true", help="ask every member every time")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--timeout", type=float, default=120, help="seconds per model call")
    p.add_argument("-v", "--verbose", action="store_true", help="show each member's answer")
    a = p.parse_args(argv)

    question = " ".join(a.question) or sys.stdin.read()
    if not question.strip():
        p.error("no question given")
    if not client.is_running():
        sys.exit(f"Ollama is not running at {client.OLLAMA_URL} — start it with `ollama serve`.")
    members = [m.strip() for m in a.members.split(",") if m.strip()]
    have = client.installed()
    missing = [m for m in members if m not in have and f"{m}:latest" not in have]
    if missing:
        sys.exit("Missing models: " + ", ".join(missing) + "\nInstall with: "
                 + " && ".join(f"ollama pull {m}" for m in missing))

    council = Council(members=members, strategy="full" if a.full else "adaptive",
                      cache_hours=0 if a.no_cache else 6, timeout=a.timeout)
    res = council.ask(question, system=a.system, schema="json" if a.json else None)
    if a.verbose:
        for m, ans in res.members.items():
            print(f"\033[2m── {m}\n{ans}\033[0m\n", file=sys.stderr)
    print(res.answer)
    print(f"\033[2m[{res.mode} · agreement {res.agreement:.2f} · {res.secs}s"
          + (" · injection flagged" if res.flagged_injection else "")
          + (f" · {res.redacted} secret(s) redacted" if res.redacted else "") + "]\033[0m",
          file=sys.stderr)
    if res.mode == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
