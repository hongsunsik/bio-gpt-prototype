"""답변 검사 (verify 노드, SFR-005·SFR-006).

줄마다 네 가지를 본다. 앞의 셋은 코드로, 마지막 하나만 LLM(검사관)으로 확인한다.
  1) 인용이 있고, 그 ID가 실제 검색된 문서인가
  2) 줄에 나온 숫자가 인용한 문서 원문에 있는가
  3) 한글·영문이 아닌 문자(키릴, 한자 등)가 섞이지 않았는가
  4) 문장 내용이 인용 문서와 맞는가 (LLM 판정)
"""
import os
import re
import time

from ..llm import chat_json
from ..models import Doc
from . import prompts as P
from .citations import BRACKET_RE, CITE_RE, ID_RE, NOT_FOUND, as_bullets, claim_lines
from .glossary import FOREIGN_SCRIPT_RE
from .policy import compliance_flags
from .state import State, add_trace

NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")
# separate: 독립 프롬프트(기본) / continue: 작성 대화에 이어 붙여 캐시 재사용.
# continue는 eval/verifier_test.py에서 더 느려서 쓰지 않는다.
VERIFY_MODE = os.getenv("VERIFY_MODE", "separate")


def line_ok(row: dict) -> bool:
    c = row["checks"]
    if c.get("not_found_ok"):
        return True
    return bool(c.get("has_citation") and c.get("citation_exists") and c.get("clean_script", True)
                and c.get("numbers_match", True) and c.get("supported", True))


# ---------------------------------------------------------------- 규칙 검사 (1~3)
def _missing_numbers(line: str, cited_text: str) -> list[str]:
    """두 자리 이상 숫자 중 인용 문서에 없는 것. 출처 ID 속 숫자는 뺀다."""
    body = ID_RE.sub("", CITE_RE.sub("", line))
    nums = [n for n in NUM_RE.findall(body) if len(n.replace(",", "").replace(".", "")) > 1]
    return [n for n in nums if n not in cited_text and n.replace(",", "") not in cited_text]


def check_line(n: int, line: str, by_id: dict[str, Doc]) -> tuple[dict, list[str]]:
    """규칙 검사 결과(row)와 작성 모델에게 돌려줄 지적(issues)."""
    cites = BRACKET_RE.findall(line)  # 형식이 틀린 인용도 인용으로 세고, 없는 ID로 처리한다
    row = {"n": n, "line": line, "cites": cites, "checks": {}}
    checks = row["checks"]
    if NOT_FOUND in line and not cites:
        checks["not_found_ok"] = True
        return row, []

    issues = []
    fake = [c for c in cites if c not in by_id]
    checks["has_citation"] = bool(cites)
    checks["citation_exists"] = not fake
    if not cites:
        issues.append(f"{n}번 줄에 인용이 없음: {line}")
    if fake:
        issues.append(f"{n}번 줄의 인용 {fake}은 근거 문서 목록에 없음. [인용 가능한 ID]의 ID만 그대로 쓸 것")

    cited_text = " ".join(f"{c} {by_id[c].text}" for c in cites if c in by_id)
    missing = _missing_numbers(line, cited_text)
    checks["numbers_match"] = not missing
    if missing and cites and not fake:
        issues.append(f"{n}번 줄의 숫자 {missing}가 인용 문서에 없음")

    foreign = FOREIGN_SCRIPT_RE.findall(line)
    checks["clean_script"] = not foreign
    if foreign:
        issues.append(f"{n}번 줄에 한글·영문이 아닌 문자 {foreign}가 섞임. 영문 일반명이나 한국어 표준 용어로 고칠 것")
    return row, issues


# ---------------------------------------------------------------- LLM 판정 (4)
def _judge_prompt(claims: list[tuple[int, str, list]], by_id: dict[str, Doc]) -> str:
    """문서는 한 번씩만 넣고 주장마다 근거 ID를 달아 입력 길이를 줄인다."""
    claim_block = "\n".join(f"[주장 {n}] (근거: {', '.join(cites)}) {CITE_RE.sub('', line).strip()}"
                            for n, line, cites in claims)
    used = list(dict.fromkeys(c for _, _, cites in claims for c in cites))
    doc_block = "\n\n".join(f"<doc id=\"{c}\">\n{by_id[c].text}\n</doc>" for c in used)
    return f"[근거 문서]\n{doc_block}\n\n[주장 목록]\n{claim_block}"


def judge_claims(claims: list[tuple[int, str, list]], by_id: dict[str, Doc], state: State) -> dict[int, str] | None:
    """근거와 맞지 않는 주장 {번호: 이유}. 검사관 응답이 깨지면 None (판정 생략)."""
    try:
        if VERIFY_MODE == "continue" and state.get("gen_messages"):
            claim_block = "\n".join(f"[주장 {n}] (근거: {', '.join(cites)}) {CITE_RE.sub('', line).strip()}"
                                    for n, line, cites in claims)
            msgs = state["gen_messages"] + [{"role": "assistant", "content": state["answer"]},
                                            {"role": "user", "content": P.VERIFIER_FOLLOWUP.format(claims=claim_block)}]
            out = chat_json("", "", messages=msgs)
        else:
            out = chat_json(P.VERIFIER_SYSTEM, P.VERIFIER_USER.format(claims=_judge_prompt(claims, by_id)), judge=True)
        return {int(r["n"]): r.get("reason", "") for r in out.get("unsupported", [])
                if isinstance(r, dict) and str(r.get("n", "")).isdigit()}
    except (ValueError, KeyError, TypeError):
        return None


# ---------------------------------------------------------------- 노드
def verify(state: State) -> State:
    t0 = time.time()
    by_id = {d.id: d for d in state["docs"]}
    passed_before = set(state.get("passed_lines", []))
    rows, issues, to_judge = [], [], []

    for n, line in enumerate(claim_lines(state["answer"]), 1):
        row, line_issues = check_line(n, line, by_id)
        rows.append(row)
        issues += line_issues
        if row["checks"].get("not_found_ok"):
            continue
        if line in passed_before:
            row["checks"]["supported"] = True  # 이전 회차에서 통과한 줄은 검사관에게 다시 보내지 않는다
        elif row["cites"] and row["checks"]["citation_exists"]:
            to_judge.append((n, line, row["cites"]))

    if to_judge:
        unsupported = judge_claims(to_judge, by_id, state)
        if unsupported is not None:
            judged = {n for n, _, _ in to_judge}
            for row in rows:
                if row["n"] not in judged:
                    continue
                row["checks"]["supported"] = row["n"] not in unsupported
                if row["n"] in unsupported:
                    row["reason"] = unsupported[row["n"]]
                    issues.append(f"{row['n']}번 줄이 근거와 불일치: {unsupported[row['n']]}")

    flags = compliance_flags(state["answer"])
    if flags:
        issues.append(f"규제 저촉 소지 표현 포함: {flags}")

    verification = {"lines": rows, "issues": issues, "compliance_flags": flags, "passed": not issues}
    runs = state.get("verify_runs", []) + [{"lines": len(rows), "bad": sum(not line_ok(r) for r in rows)}]
    return {"issues": issues, "verification": verification, "verify_runs": runs,
            "trace": add_trace(state, "verify", t0, "통과" if not issues else f"문제 {len(issues)}건")}


def verify_answer(answer: str, docs: list[Doc]) -> dict:
    """실험용: 앱과 같은 검사를 한 번 돌려 걸린 줄 수를 센다. 글머리표가 없는 서술형 답은 줄마다 주장으로 본다."""
    if not claim_lines(answer):
        answer = as_bullets(answer)
    rows = verify({"answer": answer, "docs": docs, "passed_lines": [], "verify_runs": [], "trace": []})["verification"]["lines"]
    return {"lines": len(rows), "bad": sum(not line_ok(r) for r in rows),
            "uncited": sum(not r["checks"].get("has_citation") and not r["checks"].get("not_found_ok") for r in rows)}
