"""Bio-GPT 1차년도 프로토타입 UI.  실행: uv run streamlit run app.py"""
import os
import tempfile
import time
from pathlib import Path

import streamlit as st

from bio_gpt import rag_module, store
from bio_gpt.graph import ask_stream
from bio_gpt.llm import BASE_URL, JUDGE_MODEL, MODEL
from bio_gpt.prompts import DOC_STYLES, PROMPT_VERSION

SAMPLE_PDF = Path(__file__).parent / "samples" / "keytruda_label.pdf"

st.set_page_config(page_title="Bio-GPT 프로토타입", page_icon="🧬", layout="wide")

STATUS_LABEL = {
    "answered": "✅ 검증 통과", "partial": "⚠️ 일부 문장 제외", "no_evidence": "🔍 근거 없음",
    "withheld": "⛔ 답변 보류", "refused": "🚫 응답 제한",
}
SOURCE_LABEL = {"pubmed": "논문(PubMed)", "trials": "임상(ClinicalTrials.gov)", "fda": "인허가(openFDA)", "pdf": "업로드 문서"}

if "turns" not in st.session_state:
    st.session_state.turns = []  # [{"q", "state", "qid"}]

with st.sidebar:
    st.header("🧬 Bio-GPT")
    st.caption("생성형 AI 기반 바이오·제약 R&D 지식 플랫폼 — 1차년도 핵심기능 프로토타입")
    st.markdown(f"**작성 모델** `{MODEL}`  \n**검사 모델** `{JUDGE_MODEL}`  \n**엔드포인트** `{BASE_URL}`  \n**프롬프트** `{PROMPT_VERSION}`")
    st.divider()
    s = store.stats()
    c1, c2 = st.columns(2)
    c1.metric("누적 질의", s["queries"])
    c2.metric("평균 지연", f"{s['avg_latency_ms'] / 1000:.1f}s")
    c1.metric("👍", s["likes"])
    c2.metric("👎", s["dislikes"])
    st.divider()
    # ---------------------------------------------------------- 업로드 문서 RAG (2주차)
    st.subheader("📄 문서 RAG")
    up = st.file_uploader("PDF 업로드", type=["pdf"])
    # 실행 스크립트(scripts/start_mac.command)는 BIOGPT_SAMPLE=1로 켜서 샘플 문서가 미리 열린 상태로 시작한다
    use_sample = st.checkbox("샘플: 키트루다 FDA 허가 라벨(106쪽)", value=os.getenv("BIOGPT_SAMPLE") == "1",
                             disabled=up is not None)
    with st.expander("검색 파라미터", expanded=True):
        d = rag_module.DEFAULTS
        chunk_size = st.slider("chunk_size (청크 글자 수)", 200, 2000, d["chunk_size"], 100)
        chunk_overlap = st.slider("chunk_overlap (앞뒤 청크 겹침)", 0, chunk_size // 2, min(d["chunk_overlap"], chunk_size // 2), 25)
        k = st.slider("k (가져올 청크 수)", 1, 10, d["k"])
        types = ["similarity", "mmr", "hybrid"]
        search_type = st.radio("검색 방식", types, index=types.index(d["search_type"]), horizontal=True,
                               help="similarity: 의미가 가장 가까운 k개 / mmr: 가까우면서 서로 겹치지 않는 k개 / "
                                    "hybrid: 의미 검색 + 키워드 검색(BM25)을 합침")
        translate = st.checkbox("질문을 영어 검색어로 바꿔 검색", value=d.get("translate", False),
                                help="영어 문서에서 약물·시험 이름 같은 키워드를 정확히 찾기 위해 LLM이 질문을 번역합니다")
        prompt_style = st.selectbox("프롬프트 기법", list(DOC_STYLES), index=list(DOC_STYLES).index(rag_module.DEFAULT_STYLE),
                                    format_func=DOC_STYLES.get)
        doc_only = st.checkbox("업로드 문서만 검색", value=True, help="끄면 논문·임상·FDA 검색 결과와 함께 답합니다")
    st.session_state.doc_index = None
    if up is not None or use_sample:
        if up is not None:
            path = Path(tempfile.gettempdir()) / f"biogpt_{up.name}"
            path.write_bytes(up.getbuffer())
            name = up.name
        else:
            path, name = SAMPLE_PDF, SAMPLE_PDF.name
        t0 = time.time()
        with st.spinner("문서를 나누고 임베딩하는 중… (처음 한 번만)"):
            ix = rag_module.build_index(str(path), name=name, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        st.session_state.doc_index = ix
        st.success(f"{ix.name} · {ix.pages}쪽 → 청크 {ix.chunks:,}개 ({time.time() - t0:.1f}초)")
        st.caption(f"임베딩 `{rag_module.EMBED_MODEL}` · 벡터 DB `FAISS`")
    st.divider()
    st.markdown("**예시 질문**")
    examples = [
        "면역 매개 대장염 발생률과, 그중 전신 코르티코스테로이드가 필요했던 비율은?",
        "KEYNOTE-054 임상시험에서 무재발 생존(RFS)의 위험비(HR)는?",
        "키트루다의 한국 식약처 허가 날짜는 언제인가요?",
        "키트루다의 금기 사항은 무엇인가요?",
    ] if st.session_state.doc_index else [
        "펨브롤리주맙의 FDA 허가 적응증은?",
        "소토라십 비소세포폐암 임상시험 진행 현황",
        "GLP-1 작용제와 알츠하이머병 관련 연구 결과",
        "저 당뇨인데 메트포르민 몇 mg 먹어야 해요?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.pending = ex
    if st.button("대화 초기화", type="secondary"):
        st.session_state.turns = []
        st.rerun()


def render_turn(i: int, turn: dict) -> None:
    state = turn["state"]
    with st.chat_message("user"):
        st.write(turn["q"])
    with st.chat_message("assistant", avatar="🧬"):
        status = state.get("status", "")
        mods = ", ".join(SOURCE_LABEL.get(m, m) for m in state.get("modules", [])) or "-"
        if turn.get("rag"):
            mods += f" · {turn['rag']}"
        kind = "복합 질의" if state.get("is_complex") else "단순 질의"
        st.caption(f"{STATUS_LABEL.get(status, status)} · {kind} · 모듈: {mods} · {state['latency_ms'] / 1000:.1f}초")
        st.markdown(state["final"])

        docs = state.get("docs", [])
        if docs:
            with st.expander(f"📚 근거 문서 {len(docs)}건"):
                for d in docs:
                    title = f"[{d.title}]({d.url})" if d.url else d.title
                    extra = f" · 거리 {d.meta['distance']}" if d.meta.get("distance") is not None else ""
                    st.markdown(f"**[{d.id}]** {title}  \n<small>{SOURCE_LABEL[d.source]}{extra}</small>",
                                unsafe_allow_html=True)
                    st.caption(d.text[:400] + ("…" if len(d.text) > 400 else ""))
        if state.get("reasoning"):
            with st.expander("🧠 판단 과정 (Chain of Thought)"):
                st.markdown(state["reasoning"])
        v = state.get("verification")
        if v:
            with st.expander("🔎 답변 검증 결과"):
                for row in v["lines"]:
                    c = row["checks"]
                    mark = lambda k: {True: "✅", False: "❌", None: "–"}[c.get(k)]
                    st.markdown(f"{row['n']}. {row['line']}  \n"
                                f"<small>인용 {mark('has_citation')} · 인용 실재 {mark('citation_exists')} · "
                                f"숫자 일치 {mark('numbers_match')} · 사실 판정 {mark('supported')} "
                                f"{row.get('reason', '')}</small>", unsafe_allow_html=True)
                if v["compliance_flags"]:
                    st.warning(f"규제 표현 감지: {v['compliance_flags']}")
        with st.expander("⚙️ 처리 과정"):
            ents = state.get("entities") or []
            if ents:
                st.markdown("**인식된 개체** " + ", ".join(f"`{e.get('text')}`→`{e.get('normalized')}` ({e.get('type')})" for e in ents))
            if state.get("queries"):
                st.json(state["queries"])
            for t in state["trace"]:
                st.markdown(f"- `{t['step']}` {t['ms'] / 1000:.1f}s — {t['detail']}")
            if state.get("source_errors"):
                st.error(f"소스 오류(나머지 소스로 응답): {state['source_errors']}")

        if turn.get("rated") is None:
            c1, c2, c3 = st.columns([1, 1, 6])
            comment = c3.text_input("코멘트(선택)", key=f"cm{i}", label_visibility="collapsed", placeholder="코멘트(선택)")
            if c1.button("👍", key=f"up{i}"):
                store.log_feedback(turn["qid"], 1, comment)
                turn["rated"] = 1
                st.rerun()
            if c2.button("👎", key=f"dn{i}"):
                store.log_feedback(turn["qid"], -1, comment)
                turn["rated"] = -1
                st.rerun()
        else:
            st.caption("피드백 감사합니다 " + ("👍" if turn["rated"] == 1 else "👎"))


st.title("Bio-GPT")
st.caption("논문 · 임상시험 · FDA 허가 정보를 근거로 답하고, 모든 문장에 출처를 답니다.")

for i, turn in enumerate(st.session_state.turns):
    render_turn(i, turn)

question = st.chat_input("바이오·제약 R&D 질문을 입력하세요") or st.session_state.pop("pending", None)
if question:
    history = [(t["q"], t["state"].get("final", "")) for t in st.session_state.turns]
    STEP_LABEL = {
        "guard": "보안·사용 범위 점검", "plan": "질의 분석·검색 모듈 선택", "retrieve": "문서 검색",
        "generate": "근거 기반 답변 작성", "verify": "답변 검증", "finalize": "검증 결과 반영",
        "refuse": "응답 제한", "no_evidence": "근거 없음 처리",
    }
    with st.chat_message("user"):
        st.write(question)
    with st.status("질의 분석 중…", expanded=True) as box:
        ix = st.session_state.doc_index
        options = {"doc_index": ix, "doc_only": doc_only, "prompt_style": prompt_style,
                   "rag": {"k": k, "search_type": search_type, "translate": translate}} if ix else {}
        for step, state in ask_stream(question, history, options):
            if step == "done":
                break
            t = state["trace"][-1]
            box.write(f"✔ {STEP_LABEL.get(step, step)} — {t['ms'] / 1000:.1f}초 · {t['detail']}")
            nxt = {"plan": "검색 중…", "retrieve": "답변 작성 중…", "generate": "답변 검증 중…",
                   "verify": "마무리 중…"}.get(step, "처리 중…")
            box.update(label=nxt)
        box.update(label=f"완료 ({state['latency_ms'] / 1000:.1f}초)", state="complete", expanded=False)
    qid = store.log_query(state)
    rag_label = (f"chunk {ix.chunk_size}/{ix.chunk_overlap} · k={k} · {search_type}{' · 영어 검색' if translate else ''}"
                 f" · {DOC_STYLES[prompt_style]}" if ix else "")
    st.session_state.turns.append({"q": question, "state": state, "qid": qid, "rated": None, "rag": rag_label})
    st.rerun()
