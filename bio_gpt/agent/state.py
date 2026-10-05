"""그래프 노드들이 주고받는 상태. 노드는 바꾼 키만 돌려주고 LangGraph가 합친다."""
import time
from typing import TypedDict


class State(TypedDict, total=False):
    # 입력
    question: str
    history: list  # [(질문, 답변), ...]
    doc_index: object  # rag.DocIndex | None
    doc_only: bool  # True면 업로드 문서만 검색
    rag: dict  # {"k", "search_type", "translate"}
    prompt_style: str  # prompts.DOC_STYLES의 키
    # 계획
    category: str  # research | personal_medical | off_topic | blocked
    blocked_reason: str
    modules: list
    queries: dict
    fda_sections: list
    entities: list
    rewritten: str
    # 검색
    docs: list  # list[Doc]
    source_errors: dict
    # 작성·검사
    answer: str
    reasoning: str  # CoT 기법일 때 생각 과정
    gen_messages: list  # 마지막 작성 호출의 입력 (VERIFY_MODE=continue 실험용)
    attempts: int
    issues: list
    verification: dict
    passed_lines: list  # 이전 검사를 통과한 줄. 고쳐 쓴 뒤 다시 검사하지 않는다
    verify_runs: list  # 회차별 {"lines", "bad"}. 첫 작성 기준 오류율을 잴 때 쓴다
    # 출력
    final: str
    status: str  # answered | partial | no_evidence | withheld | refused
    trace: list  # [{"step", "ms", "detail"}]


def add_trace(state: State, step: str, started: float, detail: str = "") -> list:
    """단계별 소요 시간 기록. 화면의 진행 표시와 로그에 쓴다."""
    return state.get("trace", []) + [{"step": step, "ms": int((time.time() - started) * 1000), "detail": detail}]
