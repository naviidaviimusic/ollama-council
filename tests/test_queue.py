import os

from ollama_council import queue


def test_slot_reentrant(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "QUEUE_DIR", tmp_path / "q")
    with queue.slot(max_wait=5) as a:
        assert a
        with queue.slot(max_wait=1) as b:
            assert b
    assert list((tmp_path / "q").iterdir()) == []


def test_dead_ticket_cleaned(tmp_path, monkeypatch):
    q = tmp_path / "q"
    q.mkdir()
    (q / "0.000001-999999").touch()          # pid that doesn't exist
    monkeypatch.setattr(queue, "QUEUE_DIR", q)
    with queue.slot(max_wait=5) as got:
        assert got
    assert not (q / "0.000001-999999").exists() or os.name == "nt"
