"""처리 흐름 조립 (LangGraph).

  guard → plan → retrieve → generate → verify ─┬→ finalize
     │       │        │          ↑              │
     └───────┴→ refuse └→ no_evidence └── 고쳐 쓰기 ┘ (최대 1번)

각 단계는 별도 모듈에 있고, 이 파일은 연결과 분기 조건만 갖는다.
"""
import time

from langgraph.graph import END, StateGraph

from .planner import plan
from .policy import guard
from .respond import finalize, no_evidence, refuse
from .retriever import retrieve
from .state import State
from .verifier import verify
from .writer import generate

MAX_ATTEMPTS = 2  # 첫 작성 + 고쳐 쓰기 1번
BLOCKED = ("blocked", "personal_medical")


def after_guard(state: State) -> str:
    return "refuse" if state["category"] in BLOCKED else "plan"


def after_plan(state: State) -> str:
    return "retrieve" if state["category"] == "research" else "refuse"


def after_retrieve(state: State) -> str:
    return "generate" if state["docs"] else "no_evidence"


def after_verify(state: State) -> str:
    if state["verification"]["passed"] or state["attempts"] >= MAX_ATTEMPTS:
        return "finalize"
    return "generate"


def build_graph():
    g = StateGraph(State)
    nodes = {"guard": guard, "plan": plan, "retrieve": retrieve, "generate": generate, "verify": verify,
             "finalize": finalize, "no_evidence": no_evidence, "refuse": refuse}
    for name, fn in nodes.items():
        g.add_node(name, fn)
    g.set_entry_point("guard")
    g.add_conditional_edges("guard", after_guard, ["plan", "refuse"])
    g.add_conditional_edges("plan", after_plan, ["retrieve", "refuse"])
    g.add_conditional_edges("retrieve", after_retrieve, ["generate", "no_evidence"])
    g.add_edge("generate", "verify")
    g.add_conditional_edges("verify", after_verify, ["generate", "finalize"])
    for terminal in ("finalize", "no_evidence", "refuse"):
        g.add_edge(terminal, END)
    return g.compile()


GRAPH = build_graph()


def _initial_state(question: str, history: list | None, options: dict | None) -> State:
    return {"question": question, "history": history or [], "trace": [], **(options or {})}


def _add_metrics(state: State, started: float) -> State:
    state["latency_ms"] = int((time.time() - started) * 1000)
    # PER-002 측정 조건: 검색 모듈을 2개 이상 부른 질의를 '복합 질의'로 본다
    state["is_complex"] = len(state.get("modules", [])) >= 2
    return state


def ask(question: str, history: list | None = None, options: dict | None = None) -> State:
    """options: 업로드 문서 설정 {doc_index, doc_only, rag, prompt_style}"""
    t0 = time.time()
    return _add_metrics(GRAPH.invoke(_initial_state(question, history, options)), t0)


def ask_stream(question: str, history: list | None = None, options: dict | None = None):
    """단계가 끝날 때마다 (단계 이름, 상태)를 내보낸다. 마지막은 ("done", 최종 상태)."""
    t0 = time.time()
    state: State = {}
    for state in GRAPH.stream(_initial_state(question, history, options), stream_mode="values"):
        if state.get("trace"):
            yield state["trace"][-1]["step"], state
    yield "done", _add_metrics(state, t0)
