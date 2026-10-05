"""사용자에게 나갈 최종 문구를 만든다. 면책·거절 문구는 LLM이 쓰지 않고 prompts.py의 고정 문장을 쓴다."""
import time

from . import prompts as P
from .state import State, add_trace
from .verifier import line_ok

REFUSALS = {"personal_medical": P.REFUSAL_PERSONAL, "off_topic": P.REFUSAL_OFF_TOPIC}
PARTIAL_NOTE = "\n\n_(근거 검증을 통과하지 못한 내용은 제외했습니다.)_"


def _decide(v: dict, answer: str) -> tuple[str, str]:
    """검사 결과로 (상태, 본문)을 정한다.
    통과 → 그대로 / 일부 실패 → 통과한 줄만 / 남는 줄이 없거나 규제 표현 → 답변 보류"""
    if not v["lines"]:
        return "withheld", P.WITHHELD
    if v["passed"]:
        only_not_found = all(r["checks"].get("not_found_ok") for r in v["lines"])
        return ("no_evidence" if only_not_found else "answered"), answer
    kept = [r["line"] for r in v["lines"] if line_ok(r) and not r["checks"].get("not_found_ok")]
    if kept and not v["compliance_flags"]:
        return "partial", "\n".join(kept) + PARTIAL_NOTE
    return "withheld", P.WITHHELD


def finalize(state: State) -> State:
    t0 = time.time()
    status, body = _decide(state["verification"], state["answer"])
    disclaimer = P.DISCLAIMER_DOC if state.get("modules") == ["pdf"] else P.DISCLAIMER
    return {"final": f"{body}\n\n---\n{disclaimer}", "status": status, "trace": add_trace(state, "finalize", t0, status)}


def no_evidence(state: State) -> State:
    return {"final": f"{P.NO_EVIDENCE}\n\n---\n{P.DISCLAIMER}", "status": "no_evidence",
            "trace": add_trace(state, "no_evidence", time.time(), "검색 결과 0건")}


def refuse(state: State) -> State:
    msg = REFUSALS.get(state["category"], P.REFUSAL_INJECTION)
    return {"final": msg, "status": "refused", "trace": add_trace(state, "refuse", time.time(), state["category"])}
