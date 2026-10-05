"""답변 작성 (generate 노드).

경우가 셋이다.
  1) 검사에서 걸린 줄이 있으면 그 줄만 고쳐 쓴다 (repair).
  2) 업로드 문서 RAG에서 다른 프롬프트 기법을 골랐으면 그 기법으로 쓴다 (answer_from_docs).
  3) 그 밖에는 기본 프롬프트(v1.5)로 쓴다. 규제 표현이 걸려 다시 쓰는 경우도 여기로 온다.
"""
import re
import time

from ..llm import chat_json, complete
from ..models import Doc
from . import prompts as P
from .citations import BRACKET_RE, CITE_RE, format_context, id_list, is_runaway, join_wrapped, repair_citations
from .glossary import normalize, terms_in
from .state import State, add_trace
from .verifier import line_ok

DEFAULT_STYLE = "v1.5"
RETRY_TEMPERATURE = 0.3  # 반복 폭주 후 다시 쓸 때 약간의 무작위성
INSPECTOR_WORDS = r"주장(에서|은|이)|근거 문서(에는|상)|문서상"  # 검사관 지적문에만 나오는 말투


def clean_answer(text: str, docs: list[Doc]) -> str:
    """비표준 용어 교정 → 끊긴 줄 잇기 → 인용 형식 교정."""
    return repair_citations(join_wrapped(normalize(text)), docs)


def glossary_block(docs: list[Doc]) -> str:
    terms = terms_in(" ".join(d.text for d in docs))
    return "\n".join(f"- {en} → {ko}" for en, ko in terms.items()) or "(해당 없음)"


# ---------------------------------------------------------------- 1) 걸린 줄만 고쳐 쓰기
def echoes_reason(line: str, reason: str) -> bool:
    """고친 줄이 검사관 지적을 그대로 옮겨 적었는지. 지적문의 12자 구절이 들어 있거나 검사관 말투면 그렇다고 본다."""
    if re.search(INSPECTOR_WORDS, line):
        return True
    reason_flat = re.sub(r"\s+", "", reason)
    line_flat = re.sub(r"\s+", "", line)
    return any(reason_flat[i:i + 12] in line_flat for i in range(0, max(len(reason_flat) - 11, 0), 4))


def _request_fixes(bad_rows: list[dict], docs: list[Doc]) -> dict[int, str]:
    bad_text = "\n".join(f"[{r['n']}] {r['line']}\n    지적: {r.get('reason') or '인용·숫자·글자 검사 불합격'}"
                         for r in bad_rows)
    out = chat_json(P.REPAIR_SYSTEM, P.REPAIR_USER.format(bad=bad_text, context=format_context(docs), ids=id_list(docs)))
    return {int(f["n"]): (f.get("line") or "").strip() for f in out.get("fixed", [])
            if isinstance(f, dict) and str(f.get("n", "")).isdigit()}


def repair(rows: list[dict], docs: list[Doc]) -> str:
    """통과한 줄은 그대로 두고, 걸린 줄은 고친 줄로 바꿔 원래 순서대로 합친다.
    고친 줄이 지적문을 베꼈거나 출처가 없으면 버린다."""
    bad = [r for r in rows if not line_ok(r)]
    fixed = _request_fixes(bad, docs)
    reasons = {r["n"]: r.get("reason") or "" for r in bad}
    lines = []
    for r in rows:
        if line_ok(r):
            lines.append(r["line"])
            continue
        new = fixed.get(r["n"])
        if not new or echoes_reason(new, reasons.get(r["n"], "")) or not BRACKET_RE.search(new):
            continue
        lines.append(new if new.startswith("-") else f"- {new}")
    return "\n".join(lines)


# ---------------------------------------------------------------- 2) 프롬프트 기법별 작성
def _split_cot(raw: str) -> tuple[str, str]:
    """CoT 출력에서 (생각 과정, 답변)을 나눈다. '### 답변' 표시가 없으면 인용 달린 글머리표 줄만 답으로 본다."""
    parts = re.split(r"#+\s*답변\s*\n", raw, maxsplit=1)
    if len(parts) == 2:
        return parts[0].replace("### 생각 과정", "").strip(), parts[1]
    answer = "\n".join(l for l in raw.splitlines() if l.strip().startswith("-") and CITE_RE.search(l))
    return raw, answer


def answer_from_docs(question: str, docs: list[Doc], style: str) -> dict:
    """앱과 실험 스크립트(eval/rag_sweep.py)가 같이 쓴다."""
    messages = P.doc_messages(style, question, glossary_block(docs), format_context(docs), id_list(docs))
    raw = complete(messages)
    runaway = is_runaway(raw)
    if runaway:
        raw = complete(messages, temperature=RETRY_TEMPERATURE)
    reasoning, answer = _split_cot(raw) if "cot" in style else ("", raw)
    answer = clean_answer(answer.strip(), docs)
    leak = "fewshot" in style and any(m in answer for m in P.FEWSHOT_LEAK_MARKERS)
    if is_runaway(answer):
        answer = ""  # 다시 써도 폭주하면 버린다. 검사 단계에서 답변 보류로 처리된다
    return {"answer": answer, "reasoning": reasoning, "messages": messages, "leak": leak, "runaway": runaway,
            "input_chars": sum(len(m["content"]) for m in messages)}


# ---------------------------------------------------------------- 3) 기본 작성
def _default_messages(state: State) -> list[dict]:
    feedback = P.REGENERATE_FEEDBACK.format(issues="\n".join(state["issues"])) if state.get("issues") else ""
    docs = state["docs"]
    return [{"role": "system", "content": P.GENERATOR_SYSTEM},
            {"role": "user", "content": P.GENERATOR_USER.format(
                question=state["rewritten"], glossary=glossary_block(docs), context=format_context(docs),
                ids=id_list(docs), feedback=feedback)}]


def _needs_line_repair(state: State) -> bool:
    v = state.get("verification")
    # 규제 표현은 줄 단위로 고칠 수 없어서 전체를 다시 쓴다
    return bool(v) and not v["passed"] and not v["compliance_flags"]


def generate(state: State) -> State:
    t0 = time.time()
    attempt = state.get("attempts", 0) + 1

    if _needs_line_repair(state):
        rows = state["verification"]["lines"]
        passed = [r["line"] for r in rows if line_ok(r)]
        answer = clean_answer(repair(rows, state["docs"]), state["docs"])
        return {"answer": answer, "passed_lines": passed, "attempts": attempt,
                "trace": add_trace(state, "generate", t0, f"{len(rows) - len(passed)}줄 고쳐 쓰기")}

    style = state.get("prompt_style") or DEFAULT_STYLE
    if style != DEFAULT_STYLE and not state.get("issues"):
        out = answer_from_docs(state["rewritten"], state["docs"], style)
        return {"answer": out["answer"], "reasoning": out["reasoning"], "gen_messages": out["messages"],
                "attempts": attempt, "passed_lines": [],
                "trace": add_trace(state, "generate", t0, f"{attempt}회차 · {P.DOC_STYLES[style]}")}

    messages = _default_messages(state)
    answer = clean_answer(complete(messages), state["docs"])
    return {"answer": answer, "gen_messages": messages, "attempts": attempt, "passed_lines": [],
            "trace": add_trace(state, "generate", t0, f"{attempt}회차")}
