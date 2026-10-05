"""Bio-GPT 에이전트 워크플로 (LangGraph).

  guard → plan → retrieve → generate → verify ─┬→ finalize
     │       │                   ↑              │
     └───────┴→ refuse           └── 재생성 ────┘ (검증 실패 시 1회)

SFR-001 단순화한 에이전트 구조: 1차년도는 '질의 계획 → 다중 소스 검색 → 근거 답변 → 검증'의
고정 흐름에 조건 분기만 둔다. 모듈(검색 소스)은 SOURCES 딕셔너리에 함수만 추가하면 붙는 플러그인 구조.
"""
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TypedDict

from langgraph.graph import END, StateGraph

from . import prompts as P
from .glossary import FOREIGN_SCRIPT_RE, normalize, terms_in
from .llm import chat_json, complete
from .sources import Doc, search_fda_label, search_pubmed, search_trials

# 플러그인 구조: 모듈 추가 = 여기에 한 줄 추가
SOURCES = {
    "pubmed": search_pubmed,
    "trials": search_trials,
    "fda": search_fda_label,
}
MAX_ATTEMPTS = 2  # 첫 작성 + 고쳐 쓰기 1번
# 검증 호출 방식: "separate" = 독립 프롬프트(기본) / "continue" = 생성 대화에 이어 붙임(캐시 재사용 실험, eval/verifier_test.py 결과 더 느려서 미채택)
VERIFY_MODE = os.getenv("VERIFY_MODE", "separate")

# ------------------------------------------------------------------ 규칙 기반 1차 가드 (SER-001, SFR-006)
INJECTION_PATTERNS = [
    r"(이전|위의?|모든)\s*(지시|명령|규칙).{0,6}(무시|잊)", r"ignore (all |the )?(previous|above) (instructions|prompts?)",
    r"시스템\s*프롬프트", r"system prompt", r"너의\s*(설정|지시문)", r"developer mode", r"jailbreak",
]
PERSONAL_PATTERNS = [
    r"(제가|저는|저도|내가|나는|우리\s*(엄마|아빠|아이|어머니|아버지)|저희\s*(엄마|아빠|아이|어머니|아버지))",
    r"(먹어도|복용해도|맞아도|끊어도)\s*(되|괜찮)", r"몇\s*(mg|밀리|알|정).{0,6}(먹|복용)",
]
# 의약품 광고·허가 외 사용 유도 소지가 있는 표현 (SFR-006 컴플라이언스)
COMPLIANCE_PATTERNS = [r"완치", r"부작용이?\s*(전혀\s*)?없", r"100\s*%\s*(효과|안전)", r"최고의\s*(약|치료)", r"허가\s*외.{0,6}권장"]

CITE_RE = re.compile(r"\[((?:PMID:\d+)|(?:NCT\d{8})|(?:FDA:[0-9a-f]{8}-[A-Z]+)|(?:DOC:p\d+-\d+))\]")
ID_RE = re.compile(r"PMID:?\s*\d+|NCT\d{8}|FDA:[0-9a-f]{8}-[A-Z]+|DOC:p\d+-\d+")  # 본문에 등장한 출처 ID (숫자 검증에서 제외)
BRACKET_RE = re.compile(r"\[([A-Za-z]{2,5}:? ?[^\[\]\s]{2,40})\]")  # 인용처럼 생긴 모든 괄호 (형식이 틀린 것 포함)
NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")


class State(TypedDict, total=False):
    question: str
    history: list  # [(질문, 답변), ...]
    category: str
    blocked_reason: str
    modules: list
    fda_sections: list
    queries: dict
    entities: list
    rewritten: str
    docs: list  # list[Doc]
    source_errors: dict
    answer: str
    gen_messages: list  # 마지막 생성 호출의 메시지 (검증 단계에서 캐시 재사용)
    issues: list
    verification: dict
    attempts: int
    passed_lines: list  # 검사를 통과한 줄 (고쳐 쓰기 후 다시 검사하지 않음)
    verify_runs: list  # 회차별 {"lines", "bad"} — 1차 생성 기준 환각률 측정용
    final: str
    status: str  # answered | partial | no_evidence | withheld | refused
    trace: list  # [{"step":..., "ms":..., "detail":...}]
    # 업로드 문서 RAG (2주차): 색인·검색 설정·프롬프트 기법
    doc_index: object  # rag_module.DocIndex | None
    doc_only: bool  # True면 업로드 문서만 검색 (외부 DB 검색 안 함)
    rag: dict  # {"k": 3, "search_type": "similarity"}
    prompt_style: str  # prompts.DOC_STYLES의 키. 기본 v1.5
    reasoning: str  # CoT 기법의 '생각 과정' (화면의 판단 과정 보기)


def _trace(state: State, step: str, t0: float, detail="") -> list:
    return state.get("trace", []) + [{"step": step, "ms": int((time.time() - t0) * 1000), "detail": detail}]


# ------------------------------------------------------------------ 노드
def guard(state: State) -> State:
    t0, q = time.time(), state["question"]
    if any(re.search(p, q, re.I) for p in INJECTION_PATTERNS):
        return {"category": "blocked", "blocked_reason": "injection", "trace": _trace(state, "guard", t0, "프롬프트 인젝션 패턴 차단")}
    if any(re.search(p, q) for p in PERSONAL_PATTERNS):
        return {"category": "personal_medical", "trace": _trace(state, "guard", t0, "개인 의료 상담 패턴 감지")}
    return {"category": "", "trace": _trace(state, "guard", t0, "통과")}


PERSONAL_HINT = r"(제가|저는|저도|저희|내가|나는|우리\s*(엄마|아빠|아이|애|어머니|아버지|가족)|엄마|아빠|어머니|아버지|남편|아내|와이프|할머니|할아버지|아들|딸|친구)"
DOC_POINTER = r"(이|해당|위|업로드한?)\s*(약|약물|문서|라벨|논문|임상|파일|자료)"


def _about_doc(question: str, entities: list, ix) -> bool:
    """질문이 업로드 문서에 관한 것인가. 작은 모델은 이 판단을 못 해서(실측) 규칙으로 정한다.
    ① '이 약', '이 문서' 같은 지시어 ② 질문 속 약물 이름이 문서에 나옴 ③ 약물 이름이 없으면 질환 등 다른 이름이 문서에 나옴
    ④ 이름이 하나도 없으면 문서와의 의미 거리가 가까울 때."""
    if re.search(DOC_POINTER, question):
        return True
    names = lambda types: [str(e.get("normalized") or e.get("text") or "").lower() for e in entities
                           if isinstance(e, dict) and e.get("type") in types]
    drugs, others = names(("drug",)), names(("gene", "protein", "disease", "other"))
    if drugs:
        return any(len(n) >= 3 and n in ix.text for n in drugs)
    if others:
        return any(len(n) >= 3 and n in ix.text for n in others)
    return ix.store.similarity_search_with_score(question, k=1)[0][1] < 0.75


def plan(state: State) -> State:
    t0 = time.time()
    if state.get("doc_only") and state.get("doc_index"):
        # 업로드 문서만 볼 때는 질의 분석(LLM 호출)을 건너뛴다. 다국어 임베딩(bge-m3)이라 한국어 질문으로 바로 검색한다.
        return {"category": "research", "modules": ["pdf"], "queries": {"pdf": state["question"]},
                "rewritten": state["question"], "entities": [], "trace": _trace(state, "plan", t0, "업로드 문서 검색")}
    hist = "\n".join(f"Q: {q}\nA: {a[:300]}" for q, a in state.get("history", [])[-2:]) or "(없음)"
    out = chat_json(P.PLANNER_SYSTEM, P.PLANNER_USER.format(history=hist, question=state["question"]))
    category = out.get("category", "research")
    entities = out.get("entities", []) or []
    bio = [e for e in entities if isinstance(e, dict) and e.get("type") in ("drug", "gene", "protein", "disease")]
    # 작은 모델이 과하게 거절하는 것을 규칙으로 바로잡는다 (2주차: "아스피린이 뭐야?"를 무관 질문으로 거절하던 문제)
    if category == "off_topic" and bio:
        category = "research"  # 약·질병 이름이 있으면 바이오 질문
    if category == "personal_medical" and not re.search(PERSONAL_HINT, state["question"]):
        category = "research"  # 특정 개인을 가리키는 말이 없으면 일반 정보 질문 (예: "소아 환자의 허가 용량은?")
    ix = state.get("doc_index")
    if ix and category == "research" and _about_doc(state["question"], entities, ix):
        # 업로드 문서에 관한 질문 → 문서만 검색. 문서와 무관하면 아래처럼 논문·임상·FDA 검색
        return {"category": "research", "modules": ["pdf"], "queries": {"pdf": state["question"]},
                "rewritten": state["question"], "entities": entities,
                "trace": _trace(state, "plan", t0, "업로드 문서에 관한 질문 → 문서 검색")}
    modules = [m for m in out.get("modules", []) if m in SOURCES]
    queries = out.get("queries", {}) or {}
    if category == "research" and not modules:  # 계획이 비면 논문 검색으로 기본 처리
        modules, queries = ["pubmed"], {"pubmed": out.get("rewritten_question", state["question"])}
    return {
        "category": category, "modules": modules, "queries": queries,
        "fda_sections": [x for x in out.get("fda_sections", []) or [] if x in ("IND", "DOSE", "BOX")],
        "entities": out.get("entities", []), "rewritten": out.get("rewritten_question", state["question"]),
        "trace": _trace(state, "plan", t0, f"{category} / 모듈={modules}"),
    }


def retrieve(state: State) -> State:
    t0 = time.time()
    docs, errors = [], {}

    def run(mod):
        q = state["queries"].get(mod) or state["rewritten"]
        if mod == "pdf":
            rag = state.get("rag") or {}
            return mod, state["doc_index"].search(q, k=rag.get("k", 3), search_type=rag.get("search_type", "similarity"),
                                                  translate=rag.get("translate", False))
        if mod == "fda":
            # 약물 이름을 뺀 질문 핵심어(질병명 등)로 긴 라벨에서 관련 부분을 고른다
            focus = [e.get("normalized", "") for e in state.get("entities", []) if e.get("type") != "drug"]
            return mod, SOURCES[mod](q, sections=state.get("fda_sections") or None, focus=focus)
        return mod, SOURCES[mod](q)

    with ThreadPoolExecutor(max_workers=len(state["modules"])) as ex:
        futures = {ex.submit(run, m): m for m in state["modules"]}
        for f, mod in futures.items():
            try:
                docs.extend(f.result()[1])
            except Exception as e:  # SIR-001: 한 소스가 실패해도 나머지 소스로 답변
                errors[mod] = repr(e)[:200]
    detail = f"{len(docs)}건 " + ", ".join(f"{m}:{sum(d.source == m for d in docs)}" for m in state["modules"])
    return {"docs": docs, "source_errors": errors, "attempts": 0, "issues": [], "verify_runs": [], "trace": _trace(state, "retrieve", t0, detail)}


def _context(docs: list[Doc]) -> str:
    return "\n\n".join(f"<doc id=\"{d.id}\" source=\"{d.source}\">\n{d.text}\n</doc>" for d in docs)


def _repair_citations(answer: str, docs: list[Doc]) -> str:
    """형식이 틀린 인용(예: [FDA:9333c79b-1.1])을, 가리키는 문서가 하나로 특정될 때만 진짜 ID로 고친다."""
    ids = [d.id for d in docs]
    # 괄호로 단 출처 (FDA:9333c79b-IND) → [FDA:9333c79b-IND]
    for i in ids:
        answer = answer.replace(f"({i})", f"[{i}]")

    def fix(m):
        token = m.group(1)
        if token in ids:
            return m.group(0)
        token = re.sub(r"^(PMID|FDA)\s*:?\s*", r"\1:", token)
        if token in ids:
            return f"[{token}]"
        stem = re.split(r"[-_.]", token)[0]
        cands = [i for i in ids if i.startswith(stem) or stem.endswith(i.split(":")[-1])]
        return f"[{cands[0]}]" if len(cands) == 1 else m.group(0)

    answer = BRACKET_RE.sub(fix, answer)
    # '확인되지 않음' 줄은 '근거가 없다'는 뜻이라 출처를 달 수 없다. 모델이 붙였으면 떼어 낸다
    # (붙인 채 두면 검사관이 '이 문서로는 확인되지 않음을 뒷받침 못 함'으로 걸러 맞는 답이 보류됨 — 2주차 실험 D13)
    return "\n".join(CITE_RE.sub("", l).rstrip() if "확인되지 않음" in l else l for l in answer.splitlines())


def _repair(state: State) -> str:
    """검사에서 걸린 줄만 고쳐 쓰고, 통과한 줄과 원래 순서대로 합친다 (v1.5)."""
    rows = state["verification"]["lines"]
    bad = [r for r in rows if not _line_ok(r)]
    bad_text = "\n".join(f"[{r['n']}] {r['line']}\n    지적: {r.get('reason') or '인용·숫자·글자 검사 불합격'}" for r in bad)
    out = chat_json(P.REPAIR_SYSTEM, P.REPAIR_USER.format(
        bad=bad_text, context=_context(state["docs"]), ids=", ".join(d.id for d in state["docs"])))
    fixed = {int(f["n"]): (f.get("line") or "").strip() for f in out.get("fixed", [])
             if isinstance(f, dict) and str(f.get("n", "")).isdigit()}
    reasons = {r["n"]: r.get("reason") or "" for r in bad}
    lines = []
    for r in rows:
        if _line_ok(r):
            lines.append(r["line"])
        elif fixed.get(r["n"]) and _echoes_reason(fixed[r["n"]], reasons.get(r["n"], "")):
            continue  # 고친 줄 대신 검사관의 지적문을 옮겨 적은 경우 버림 (2주차 실험 D07)
        elif fixed.get(r["n"]) and BRACKET_RE.search(fixed[r["n"]]):  # 출처 없는 고친 줄은 버림
            lines.append(fixed[r["n"]] if fixed[r["n"]].startswith("-") else f"- {fixed[r['n']]}")
    return "\n".join(lines)


def _echoes_reason(line: str, reason: str) -> bool:
    """고친 줄이 검사관 지적을 베꼈는지: 지적문의 12자 이상 구절이 그대로 있거나 '주장' 같은 검사 용어가 들어 있으면 그렇다고 본다."""
    if re.search(r"주장(에서|은|이)|근거 문서(에는|상)|문서상", line):
        return True
    text = re.sub(r"\s+", "", reason)
    flat = re.sub(r"\s+", "", line)
    return any(text[i:i + 12] in flat for i in range(0, max(len(text) - 11, 0), 4))


def generate(state: State) -> State:
    t0 = time.time()
    v = state.get("verification")
    if v and not v["passed"] and not v["compliance_flags"]:
        # 2회차: 걸린 줄만 고쳐 쓴다. 통과한 줄은 기억해 두고 다시 검사하지 않는다.
        answer = _repair(state)
        answer = _repair_citations(_join_wrapped(normalize(answer)), state["docs"])
        passed = [r["line"] for r in v["lines"] if _line_ok(r)]
        return {"answer": answer, "passed_lines": passed, "attempts": state["attempts"] + 1,
                "trace": _trace(state, "generate", t0, f"{len(v['lines']) - len(passed)}줄 고쳐 쓰기")}
    style = state.get("prompt_style") or "v1.5"
    if style != "v1.5" and not state.get("issues"):
        # 업로드 문서 RAG에서 고른 프롬프트 기법으로 작성 (2주차 실험)
        out = answer_from_docs(state["rewritten"], state["docs"], style)
        return {"answer": out["answer"], "reasoning": out["reasoning"], "gen_messages": out["messages"],
                "attempts": state.get("attempts", 0) + 1, "passed_lines": [],
                "trace": _trace(state, "generate", t0, f"{state.get('attempts', 0) + 1}회차 · {P.DOC_STYLES[style]}")}
    feedback = P.REGENERATE_FEEDBACK.format(issues="\n".join(state["issues"])) if state.get("issues") else ""
    messages = [{"role": "system", "content": P.GENERATOR_SYSTEM},
                {"role": "user", "content": P.GENERATOR_USER.format(
                    question=state["rewritten"], glossary=_glossary(state["docs"]), context=_context(state["docs"]),
                    ids=", ".join(d.id for d in state["docs"]), feedback=feedback)}]
    answer = complete(messages)
    answer = _join_wrapped(normalize(answer))  # 줄 이어 붙이기 + 비표준 표기(멜라노마 등) → 표준 용어
    answer = _repair_citations(answer, state["docs"])
    return {"answer": answer, "gen_messages": messages, "attempts": state.get("attempts", 0) + 1, "passed_lines": [],
            "trace": _trace(state, "generate", t0, f"{state.get('attempts', 0) + 1}회차")}


def _glossary(docs: list[Doc]) -> str:
    return "\n".join(f"- {en} → {ko}" for en, ko in terms_in(" ".join(d.text for d in docs)).items()) or "(해당 없음)"


def answer_from_docs(question: str, docs: list[Doc], style: str) -> dict:
    """주어진 검색 결과로 프롬프트 기법(style)에 따라 답을 쓴다. 실험(eval/rag_sweep.py)과 앱이 같이 쓴다."""
    context = _context(docs)
    messages = P.doc_messages(style, question, _glossary(docs), context, ", ".join(d.id for d in docs))
    raw = complete(messages)
    runaway = _is_runaway(raw)
    if runaway:  # 반복 폭주 → 약간의 무작위성을 줘서 한 번 다시 쓴다
        raw = complete(messages, temperature=0.3)
    reasoning, answer = "", raw
    if "cot" in style:
        # '### 답변' 앞은 생각 과정, 뒤는 답변. 표시가 없으면 글머리표 줄만 답변으로 본다.
        parts = re.split(r"#+\s*답변\s*\n", raw, maxsplit=1)
        if len(parts) == 2:
            reasoning, answer = parts[0].replace("### 생각 과정", "").strip(), parts[1]
        else:
            reasoning = raw
            answer = "\n".join(l for l in raw.splitlines() if l.strip().startswith("-") and CITE_RE.search(l))
    answer = _repair_citations(_join_wrapped(normalize(answer.strip())), docs)
    leak = "fewshot" in style and any(m in answer for m in P.FEWSHOT_LEAK_MARKERS)
    if _is_runaway(answer):
        answer = ""  # 다시 써도 폭주하면 답을 버린다 → 검증 단계에서 답변 보류
    return {"answer": answer, "reasoning": reasoning, "messages": messages, "leak": leak, "runaway": runaway,
            "input_chars": sum(len(m["content"]) for m in messages)}


def _is_runaway(text: str) -> bool:
    """같은 구절(8자 이상)이 연달아 6번 넘게 반복되면 반복 폭주로 본다."""
    return bool(re.search(r"(.{8,40}?)\1{5,}", text, flags=re.S))


def verify_answer(answer: str, docs: list[Doc]) -> dict:
    """실험용: 앱과 같은 검증(verify)을 한 번 돌려 걸린 줄 수를 센다.
    글머리표가 없는 서술형 답(기본 예제 프롬프트)은 문장 줄 전체를 주장으로 본다."""
    if not _claim_lines(answer):
        answer = "\n".join(f"- {l.strip()}" for l in answer.splitlines() if l.strip())
    v = verify({"answer": answer, "docs": docs, "passed_lines": [], "verify_runs": [], "trace": []})["verification"]
    rows = v["lines"]
    return {"lines": len(rows), "bad": sum(not _line_ok(r) for r in rows),
            "uncited": sum(not r["checks"].get("has_citation") and not r["checks"].get("not_found_ok") for r in rows)}


def _join_wrapped(answer: str) -> str:
    """AI가 한 문장을 여러 줄로 나눠 쓰면 출처가 다음 줄로 밀려 '출처 없음'으로 걸린다.
    글머리표(-)로 시작하지 않는 줄은 앞 줄에 이어 붙인다."""
    out = []
    for l in answer.splitlines():
        s = l.strip()
        if out and s and not s.startswith(("-", "•", "*", "_")) and out[-1].lstrip().startswith(("-", "•", "*")):
            out[-1] = f"{out[-1].rstrip()} {s}"
        else:
            out.append(l)
    return "\n".join(out)


def _claim_lines(answer: str) -> list[str]:
    return [l.strip() for l in answer.splitlines() if l.strip().startswith(("-", "•", "*")) or CITE_RE.search(l)]


def verify(state: State) -> State:
    """검증: ① 인용 실재성 ② 숫자 일치 ③ 외국 문자 혼입 ④ LLM 사실 판정."""
    t0 = time.time()
    by_id = {d.id: d for d in state["docs"]}
    lines = _claim_lines(state["answer"])
    issues, per_line, judge_input = [], [], []

    for i, line in enumerate(lines, 1):
        # 형식이 틀린 인용도 '인용'으로 세고, 실재하지 않는 인용으로 처리한다
        cites = [c for c in BRACKET_RE.findall(line)]
        row = {"n": i, "line": line, "cites": cites, "checks": {}}
        if "확인되지 않음" in line and not cites:
            row["checks"]["not_found_ok"] = True
            per_line.append(row)
            continue
        fake = [c for c in cites if c not in by_id]
        row["checks"]["has_citation"] = bool(cites)
        row["checks"]["citation_exists"] = not fake
        if not cites:
            issues.append(f"{i}번 줄에 인용이 없음: {line}")
        if fake:
            issues.append(f"{i}번 줄의 인용 {fake}은 근거 문서 목록에 없음. [인용 가능한 ID]의 ID만 그대로 쓸 것")
        # ② 숫자 일치: 주장의 숫자(인용 ID 제외, 한 자리 정수 제외)가 인용 문서 원문에 있어야 함
        cited_text = " ".join(f"{c} {by_id[c].text}" for c in cites if c in by_id)
        nums = [n for n in NUM_RE.findall(ID_RE.sub("", CITE_RE.sub("", line))) if len(n.replace(",", "").replace(".", "")) > 1]
        missing = [n for n in nums if n not in cited_text and n.replace(",", "") not in cited_text]
        row["checks"]["numbers_match"] = not missing
        if missing and cites and not fake:
            issues.append(f"{i}번 줄의 숫자 {missing}가 인용 문서에 없음")
        # ③ 외국 문자 혼입 (예: '페메트레ксed')
        foreign = FOREIGN_SCRIPT_RE.findall(line)
        row["checks"]["clean_script"] = not foreign
        if foreign:
            issues.append(f"{i}번 줄에 한글·영문이 아닌 문자 {foreign}가 섞임. 영문 일반명이나 한국어 표준 용어로 고칠 것")
        if line in state.get("passed_lines", []):
            row["checks"]["supported"] = True  # 이전 검사에서 통과한 줄 → AI 검사관에게 다시 보내지 않음
        elif cites and not fake:
            judge_input.append((i, line, cites))
        per_line.append(row)

    # ④ LLM 판정 (인용이 유효한 줄만). 문서는 한 번씩만 넣고 주장마다 근거 ID를 표시해 입력 길이를 줄인다.
    if judge_input:
        claim_block = "\n".join(f"[주장 {i}] (근거: {', '.join(cites)}) {CITE_RE.sub('', line).strip()}" for i, line, cites in judge_input)
        try:
            if VERIFY_MODE == "continue" and state.get("gen_messages"):
                msgs = state["gen_messages"] + [{"role": "assistant", "content": state["answer"]},
                                                {"role": "user", "content": P.VERIFIER_FOLLOWUP.format(claims=claim_block)}]
                out = chat_json("", "", messages=msgs)
            else:
                used = list(dict.fromkeys(c for _, _, cites in judge_input for c in cites))
                doc_block = "\n\n".join(f"<doc id=\"{c}\">\n{by_id[c].text}\n</doc>" for c in used)
                out = chat_json(P.VERIFIER_SYSTEM, P.VERIFIER_USER.format(
                    claims=f"[근거 문서]\n{doc_block}\n\n[주장 목록]\n{claim_block}"), judge=True)
            unsupported = {int(r["n"]): r.get("reason", "") for r in out.get("unsupported", [])
                           if isinstance(r, dict) and str(r.get("n", "")).isdigit()}
            judged = True
        except (ValueError, KeyError, TypeError):
            unsupported, judged = {}, False
        judged_ns = {i for i, _, _ in judge_input}
        for row in per_line:
            if judged and row["n"] in judged_ns:
                row["checks"]["supported"] = row["n"] not in unsupported
                if row["n"] in unsupported:
                    row["reason"] = unsupported[row["n"]]
                    issues.append(f"{row['n']}번 줄이 근거와 불일치: {unsupported[row['n']]}")

    compliance = [p for p in COMPLIANCE_PATTERNS if re.search(p, state["answer"])]
    if compliance:
        issues.append(f"규제 저촉 소지 표현 포함: {compliance}")

    verification = {"lines": per_line, "issues": issues, "compliance_flags": compliance, "passed": not issues}
    bad = sum(1 for r in per_line if not _line_ok(r))
    runs = state.get("verify_runs", []) + [{"lines": len(per_line), "bad": bad}]
    return {"issues": issues, "verification": verification, "verify_runs": runs,
            "trace": _trace(state, "verify", t0, "통과" if not issues else f"문제 {len(issues)}건")}


def _line_ok(row: dict) -> bool:
    c = row["checks"]
    return c.get("not_found_ok") or (c.get("has_citation") and c.get("citation_exists") and c.get("clean_script", True)
                                     and c.get("numbers_match", True) and c.get("supported", True))


def finalize(state: State) -> State:
    t0 = time.time()
    v = state["verification"]
    if not v["lines"]:  # 고쳐 쓰기 후 남은 줄이 없음
        status, body = "withheld", P.WITHHELD
    elif v["passed"]:
        status, body = ("no_evidence" if all(r["checks"].get("not_found_ok") for r in v["lines"]) else "answered"), state["answer"]
    else:
        # 검증 실패 줄이 적거나 재생성 후에도 실패: 통과한 줄만 남기고, 하나도 없으면 답변 보류
        kept = [r["line"] for r in v["lines"] if _line_ok(r) and not r["checks"].get("not_found_ok")]
        if kept and not v["compliance_flags"]:
            status = "partial"
            body = "\n".join(kept) + "\n\n_(근거 검증을 통과하지 못한 내용은 제외했습니다.)_"
        else:
            status, body = "withheld", P.WITHHELD
    disclaimer = P.DISCLAIMER_DOC if state.get("modules") == ["pdf"] else P.DISCLAIMER
    return {"final": f"{body}\n\n---\n{disclaimer}", "status": status, "trace": _trace(state, "finalize", t0, status)}


def no_evidence(state: State) -> State:
    return {"final": f"{P.NO_EVIDENCE}\n\n---\n{P.DISCLAIMER}", "status": "no_evidence",
            "trace": _trace(state, "no_evidence", time.time(), "검색 결과 0건")}


def refuse(state: State) -> State:
    msg = {"personal_medical": P.REFUSAL_PERSONAL, "off_topic": P.REFUSAL_OFF_TOPIC}.get(state["category"], P.REFUSAL_INJECTION)
    return {"final": msg, "status": "refused", "trace": _trace(state, "refuse", time.time(), state["category"])}


# ------------------------------------------------------------------ 분기
def after_guard(state: State) -> str:
    return "refuse" if state["category"] in ("blocked", "personal_medical") else "plan"


def after_plan(state: State) -> str:
    return "retrieve" if state["category"] == "research" else "refuse"


def after_retrieve(state: State) -> str:
    return "generate" if state["docs"] else "no_evidence"


def after_verify(state: State) -> str:
    # 통과했거나 이미 한 번 고쳐 썼으면 마무리. 아니면 다시 작성 단계로
    # (걸린 줄만 고쳐 쓰기. 규제 표현이 있으면 답 전체를 다시 씀 — generate()에서 판단)
    v = state["verification"]
    if v["passed"] or state["attempts"] >= MAX_ATTEMPTS:
        return "finalize"
    return "generate"


def build_graph():
    g = StateGraph(State)
    for name, fn in [("guard", guard), ("plan", plan), ("retrieve", retrieve), ("generate", generate),
                     ("verify", verify), ("finalize", finalize), ("no_evidence", no_evidence), ("refuse", refuse)]:
        g.add_node(name, fn)
    g.set_entry_point("guard")
    g.add_conditional_edges("guard", after_guard, ["plan", "refuse"])
    g.add_conditional_edges("plan", after_plan, ["retrieve", "refuse"])
    g.add_conditional_edges("retrieve", after_retrieve, ["generate", "no_evidence"])
    g.add_edge("generate", "verify")
    g.add_conditional_edges("verify", after_verify, ["generate", "finalize"])
    for end in ("finalize", "no_evidence", "refuse"):
        g.add_edge(end, END)
    return g.compile()


GRAPH = build_graph()


def ask_stream(question: str, history: list | None = None, options: dict | None = None):
    """단계가 끝날 때마다 (단계 이름, 상태)를 내보낸다. UI 진행 표시용. 마지막 값이 최종 상태.
    options: 업로드 문서 RAG 설정 {doc_index, doc_only, rag, prompt_style}"""
    t0 = time.time()
    state = {}
    for state in GRAPH.stream({"question": question, "history": history or [], "trace": [], **(options or {})},
                              stream_mode="values"):
        trace = state.get("trace") or []
        if trace:
            yield trace[-1]["step"], state
    state["latency_ms"] = int((time.time() - t0) * 1000)
    state["is_complex"] = len(state.get("modules", [])) >= 2
    yield "done", state


def ask(question: str, history: list | None = None, options: dict | None = None) -> State:
    t0 = time.time()
    state = GRAPH.invoke({"question": question, "history": history or [], "trace": [], **(options or {})})
    state["latency_ms"] = int((time.time() - t0) * 1000)
    # PER-002 측정 조건: '복합 질의' = 2개 이상의 검색 모듈을 호출한 질의
    state["is_complex"] = len(state.get("modules", [])) >= 2
    return state
