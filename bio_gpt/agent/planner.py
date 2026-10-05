"""질문 분석과 검색 경로 결정 (SFR-002).

LLM이 질문을 분류하고 검색어를 만든다. 작은 로컬 모델이 틀리기 쉬운 두 가지
(과잉 거절, 업로드 문서와 관련 있는지 판단)는 규칙으로 보정한다.
"""
import re
import time

from ..llm import chat_json
from ..sources import SOURCES
from . import prompts as P
from .policy import mentions_person
from .state import State, add_trace

BIO_TYPES = ("drug", "gene", "protein", "disease")
FDA_SECTIONS = ("IND", "DOSE", "BOX")
DOC_POINTER = r"(이|해당|위|업로드한?)\s*(약|약물|문서|라벨|논문|임상|파일|자료)"
DOC_DISTANCE_MAX = 0.75  # 이름이 없는 질문은 문서와의 임베딩 거리가 이보다 가까우면 문서 질문으로 본다
HISTORY_TURNS = 2


def _entity_names(entities: list, types: tuple) -> list[str]:
    return [str(e.get("normalized") or e.get("text") or "").lower()
            for e in entities if isinstance(e, dict) and e.get("type") in types]


def is_about_doc(question: str, entities: list, ix) -> bool:
    """질문이 업로드 문서에 관한 것인지. 순서대로 확인한다.
    1) '이 약', '이 문서' 같은 지시어  2) 약물 이름이 문서에 나옴
    3) 약물 이름이 없으면 질환 등 다른 이름이 문서에 나옴  4) 이름이 없으면 임베딩 거리"""
    if re.search(DOC_POINTER, question):
        return True
    drugs = _entity_names(entities, ("drug",))
    others = _entity_names(entities, ("gene", "protein", "disease", "other"))
    names = drugs or others
    if names:
        return any(len(n) >= 3 and n in ix.text for n in names)
    return ix.store.similarity_search_with_score(question, k=1)[0][1] < DOC_DISTANCE_MAX


def _correct_category(category: str, question: str, entities: list) -> str:
    """8B 모델의 과잉 거절을 되돌린다.
    '아스피린이 뭐야'를 off_topic으로, '소아 허가 용량은?'을 personal_medical로 분류하던 문제."""
    has_bio = any(isinstance(e, dict) and e.get("type") in BIO_TYPES for e in entities)
    if category == "off_topic" and has_bio:
        return "research"
    if category == "personal_medical" and not mentions_person(question):
        return "research"
    return category


def _doc_plan(state: State, entities: list, t0: float, detail: str) -> State:
    q = state["question"]
    return {"category": "research", "modules": ["pdf"], "queries": {"pdf": q}, "rewritten": q,
            "entities": entities, "trace": add_trace(state, "plan", t0, detail)}


def _format_history(history: list) -> str:
    return "\n".join(f"Q: {q}\nA: {a[:300]}" for q, a in history[-HISTORY_TURNS:]) or "(없음)"


def plan(state: State) -> State:
    t0 = time.time()
    question, ix = state["question"], state.get("doc_index")
    if state.get("doc_only") and ix:
        # 문서만 볼 때는 LLM 분석을 건너뛴다. bge-m3가 다국어라 한국어 질문 그대로 검색된다.
        return _doc_plan(state, [], t0, "업로드 문서 검색")

    out = chat_json(P.PLANNER_SYSTEM, P.PLANNER_USER.format(history=_format_history(state.get("history", [])),
                                                            question=question))
    entities = out.get("entities", []) or []
    category = _correct_category(out.get("category", "research"), question, entities)
    if ix and category == "research" and is_about_doc(question, entities, ix):
        return _doc_plan(state, entities, t0, "업로드 문서에 관한 질문 → 문서 검색")

    rewritten = out.get("rewritten_question", question)
    modules = [m for m in out.get("modules", []) if m in SOURCES]
    queries = out.get("queries", {}) or {}
    if category == "research" and not modules:  # 계획이 비면 논문 검색
        modules, queries = ["pubmed"], {"pubmed": rewritten}
    return {
        "category": category, "modules": modules, "queries": queries,
        "fda_sections": [x for x in out.get("fda_sections", []) or [] if x in FDA_SECTIONS],
        "entities": out.get("entities", []), "rewritten": rewritten,
        "trace": add_trace(state, "plan", t0, f"{category} / 모듈={modules}"),
    }
