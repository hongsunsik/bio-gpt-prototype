"""질의·응답·근거·피드백 로그 저장 (SIR-002 로그 인터페이스, SFR-007 피드백 수집)."""
import json
import sqlite3
import time

from .agent.prompts import PROMPT_VERSION
from .config import DATA_DIR, JUDGE, WRITER

DB = DATA_DIR / "bio_gpt.db"


def _conn():
    DB.parent.mkdir(exist_ok=True)
    c = sqlite3.connect(DB)
    c.execute("""CREATE TABLE IF NOT EXISTS queries (
        id INTEGER PRIMARY KEY, ts REAL, question TEXT, category TEXT, modules TEXT, is_complex INTEGER,
        status TEXT, latency_ms INTEGER, trace TEXT, answer TEXT, doc_ids TEXT, verification TEXT,
        model TEXT, prompt_version TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY, query_id INTEGER, ts REAL, rating INTEGER, comment TEXT)""")
    return c


def _model_label() -> str:
    if JUDGE.model == WRITER.model:
        return WRITER.model
    return f"{WRITER.model} (검사관 {JUDGE.model})"


def log_query(state: dict) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO queries (ts, question, category, modules, is_complex, status, latency_ms, trace, answer,"
            " doc_ids, verification, model, prompt_version) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (time.time(), state["question"], state.get("category"), json.dumps(state.get("modules", [])),
             int(state.get("is_complex", False)), state.get("status"), state.get("latency_ms"),
             json.dumps(state.get("trace", []), ensure_ascii=False), state.get("final"),
             json.dumps([d.id for d in state.get("docs", [])]),
             json.dumps(state.get("verification", {}), ensure_ascii=False),
             _model_label(), PROMPT_VERSION))
        return cur.lastrowid


def log_feedback(query_id: int, rating: int, comment: str = "") -> None:
    """rating: 1 = Like, -1 = Dislike"""
    with _conn() as c:
        c.execute("INSERT INTO feedback (query_id, ts, rating, comment) VALUES (?,?,?,?)",
                  (query_id, time.time(), rating, comment))


def stats() -> dict:
    with _conn() as c:
        q = c.execute("SELECT COUNT(*), AVG(latency_ms) FROM queries").fetchone()
        f = c.execute("SELECT SUM(rating=1), SUM(rating=-1) FROM feedback").fetchone()
        by_status = dict(c.execute("SELECT status, COUNT(*) FROM queries GROUP BY status").fetchall())
    return {"queries": q[0], "avg_latency_ms": int(q[1] or 0), "likes": f[0] or 0, "dislikes": f[1] or 0,
            "by_status": by_status}


def export_jsonl(path: str) -> int:
    """관리자 내보내기: 질의·응답·피드백을 표준 JSONL로 내보낸다."""
    with _conn() as c:
        c.row_factory = sqlite3.Row
        rows = c.execute("""SELECT q.*, GROUP_CONCAT(f.rating) AS ratings, GROUP_CONCAT(f.comment, ' | ') AS comments
                            FROM queries q LEFT JOIN feedback f ON f.query_id = q.id GROUP BY q.id""").fetchall()
    with open(path, "w") as fp:
        for r in rows:
            fp.write(json.dumps(dict(r), ensure_ascii=False) + "\n")
    return len(rows)
