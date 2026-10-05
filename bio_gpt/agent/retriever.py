"""선택된 소스를 병렬로 검색한다. 한 소스가 실패해도 나머지로 답한다 (SIR-001)."""
import time
from concurrent.futures import ThreadPoolExecutor

from ..models import Doc
from ..sources import SOURCES
from .state import State, add_trace


def _search(module: str, state: State) -> list[Doc]:
    query = state["queries"].get(module) or state["rewritten"]
    if module == "pdf":
        rag = state.get("rag") or {}
        return state["doc_index"].search(query, k=rag.get("k", 3), search_type=rag.get("search_type", "similarity"),
                                         translate=rag.get("translate", False))
    if module == "fda":
        # 약물 이름을 뺀 핵심어(질환명 등)로 긴 라벨에서 관련 부분을 고른다
        focus = [e.get("normalized", "") for e in state.get("entities", []) if e.get("type") != "drug"]
        return SOURCES["fda"](query, sections=state.get("fda_sections") or None, focus=focus)
    return SOURCES[module](query)


def retrieve(state: State) -> State:
    t0 = time.time()
    modules = state["modules"]
    docs, errors = [], {}
    with ThreadPoolExecutor(max_workers=len(modules)) as ex:  # 외부 API 대기가 대부분이라 스레드로 충분
        futures = {m: ex.submit(_search, m, state) for m in modules}
        for module, future in futures.items():
            try:
                docs.extend(future.result())
            except Exception as e:
                errors[module] = repr(e)[:200]
    detail = f"{len(docs)}건 " + ", ".join(f"{m}:{sum(d.source == m for d in docs)}" for m in modules)
    return {"docs": docs, "source_errors": errors, "attempts": 0, "issues": [], "verify_runs": [],
            "trace": add_trace(state, "retrieve", t0, detail)}
