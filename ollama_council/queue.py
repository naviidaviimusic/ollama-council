"""Cross-process FIFO queue for a local model server.

Many scripts sharing one CPU-bound Ollama is the common failure: they all fire at
once, every request slows down, all of them time out. This queue lets N
processes (default 1) talk to the model at a time, in arrival order.

Waiters take a ticket file and only try the lock while their ticket is among the
oldest `slots` tickets owned by live processes, so late arrivals can't starve
early ones. Tickets of dead processes are cleaned up automatically.
POSIX only (fcntl); on other platforms the queue is a no-op.
"""

import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path

try:
    import fcntl
except ImportError:          # Windows
    fcntl = None

QUEUE_DIR = Path(os.environ.get("COUNCIL_QUEUE_DIR",
                                Path.home() / ".cache" / "ollama-council" / "queue"))
_local = threading.local()


def _my_turn(qdir: Path, ticket: Path, slots: int) -> bool:
    live = []
    while True:
        try:
            entries = sorted(qdir.iterdir())
            break
        except InterruptedError:      # a signal interrupted the listing; retry
            continue
        except FileNotFoundError:
            return False
    for t in entries:
        try:
            os.kill(int(t.name.rsplit("-", 1)[1]), 0)
            live.append(t)
        except (ValueError, IndexError, ProcessLookupError):
            t.unlink(missing_ok=True)  # malformed, or owner died without cleaning up
        except PermissionError:
            live.append(t)
        if len(live) >= slots:
            break
    return ticket in live


@contextmanager
def slot(max_wait: float = 1800, slots: int = None):
    """Yield True once this process holds a slot, False if none freed in max_wait.
    Re-entrant within a thread, so a council holds one slot for its whole decision."""
    slots = slots or int(os.environ.get("COUNCIL_SLOTS", "1"))
    if fcntl is None or getattr(_local, "held", False):
        yield True
        return
    qdir = QUEUE_DIR
    qdir.mkdir(parents=True, exist_ok=True)
    ticket = qdir / f"{time.time():.6f}-{os.getpid()}"
    ticket.touch()
    give_up = time.monotonic() + max_wait
    held = None
    try:
        while held is None:
            if _my_turn(qdir, ticket, slots):
                for i in range(slots):
                    fh = open(qdir.parent / f"slot{i}.lock", "w")
                    try:
                        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        held = fh
                        break
                    except BlockingIOError:
                        fh.close()
            if held is None:
                if time.monotonic() > give_up:
                    break
                time.sleep(0.5)
    finally:
        ticket.unlink(missing_ok=True)   # kept until the lock is held, so no one jumps in
    if held is None:
        yield False
        return
    try:
        _local.held = True
        yield True
    finally:
        _local.held = False
        fcntl.flock(held, fcntl.LOCK_UN)
        held.close()
