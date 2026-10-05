"""업로드 문서 RAG 파라미터 실험 (2주차 과제: chunk_size · overlap · k · 검색 방식 · 프롬프트 비교).

  1단계 retrieval : LLM 없이 '정답 문장이 검색된 청크 안에 들어 있는가'만 본다. 빠르고 공짜라 조합을 넓게 훑는다.
  2단계 prompt    : 검색 설정을 고정하고 프롬프트 기법(기본 예제 · v1.5 · 형식 지정 · Few-shot · CoT …)만 바꿔 답을 비교한다.
  3단계 params    : 프롬프트를 고정하고 1단계에서 고른 검색 설정 몇 개로 실제 답을 비교한다.

실행:
  uv run python eval/rag_sweep.py retrieval
  uv run python eval/rag_sweep.py prompt
  uv run python eval/rag_sweep.py params
결과: eval/results/rag_<단계>_<시각>.md (+ .json)
"""
import json
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bio_gpt.rag import build_index  # noqa: E402

ROOT = Path(__file__).resolve().parent
PDF = ROOT.parent / "samples" / "keytruda_label.pdf"
ITEMS = [json.loads(l) for l in (ROOT / "pdf_golden_set.jsonl").read_text().splitlines() if l.strip()]
ITEM_BY_ID = {it["id"]: it for it in ITEMS}


def squash(s: str) -> str:
    return re.sub(r"\s+", " ", s)


def hit(docs, evidence) -> bool:
    """정답 근거(정규식)가 검색된 청크 중 하나에 통째로 들어 있으면 적중."""
    return evidence is not None and any(re.search(evidence, squash(d.text)) for d in docs)


def correct(answer: str, keywords: list) -> bool:
    """정답 키워드 묶음이 모두 들어 있으면 정답. 묶음 안의 표기 중 하나만 있으면 된다."""
    a = answer.replace(" ", " ")
    return all(any(k in a for k in group) for group in keywords)


def save(stage: str, md: str, data) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M")
    out = ROOT / "results" / f"rag_{stage}_{stamp}.md"
    out.write_text(md)
    out.with_suffix(".json").write_text(json.dumps(data, ensure_ascii=False, indent=1, default=str))
    print(f"→ {out}")
    return out


# ------------------------------------------------------------------ 1단계: 검색만
def stage_retrieval():
    answerable = [it for it in ITEMS if it["evidence"]]
    rows = []
    for size in (200, 400, 800, 1500):
        for overlap in (0, size // 4):
            t0 = time.time()
            ix = build_index(str(PDF), chunk_size=size, chunk_overlap=overlap)
            build_s = time.time() - t0
            for search_type in ("similarity", "mmr", "hybrid"):
                for k in (1, 3, 5, 8):
                    for translate in (False, True):
                        hits, ctx, missed = 0, [], []
                        for it in answerable:
                            docs = ix.search(it["question"], k=k, search_type=search_type, translate=translate)
                            ok = hit(docs, it["evidence"])
                            hits += ok
                            missed += [] if ok else [it["id"]]
                            ctx.append(sum(len(d.text) for d in docs))
                        rows.append({"chunk_size": size, "overlap": overlap, "chunks": ix.chunks, "build_s": round(build_s, 1),
                                     "search": search_type, "k": k, "query": "en(번역)" if translate else "ko",
                                     "hit": hits / len(answerable), "ctx_chars": int(statistics.mean(ctx)), "missed": missed})
            print(f"size={size} overlap={overlap} chunks={ix.chunks} build={build_s:.1f}s", flush=True)

    rows.sort(key=lambda r: (-r["hit"], r["ctx_chars"]))
    md = [f"# 1단계: 검색 파라미터 실험 (LLM 없음)\n",
          f"- 문서: `{PDF.name}` (106쪽) · 질문 {len(answerable)}개(문서에 답이 있는 것만)",
          "- **적중률** = 정답 근거 문장(evidence)이 검색된 청크 안에 통째로 들어 있는 질문 비율",
          "- **질문 언어** ko = 한국어 질문 그대로 / en(번역) = LLM이 영어 검색어로 바꿔서 검색",
          "- **검색** hybrid = 의미 검색(FAISS) + 키워드 검색(BM25)을 순위로 합침(RRF)",
          "- **입력 길이** = LLM에 넘기는 검색 결과 글자 수 평균 (토큰 비용과 응답 시간에 비례)\n",
          "| chunk_size | overlap | 청크 수 | 색인(초) | 검색 | k | 질문 언어 | 적중률 | 입력 길이(자) | 못 찾은 문항 |",
          "|---:|---:|---:|---:|---|---:|---|---:|---:|---|"]
    md += [f"| {r['chunk_size']} | {r['overlap']} | {r['chunks']} | {r['build_s']} | {r['search']} | {r['k']} | {r['query']} | "
           f"{r['hit'] * 100:.0f}% | {r['ctx_chars']:,} | {' '.join(r['missed'])} |" for r in rows]
    save("retrieval", "\n".join(md), rows)


# ------------------------------------------------------------------ 2·3단계: 실제 답 생성
def run_answers(style: str, size: int, overlap: int, k: int, search_type: str, query: str = "ko") -> dict:
    from bio_gpt.agent.verifier import verify_answer
    from bio_gpt.agent.writer import answer_from_docs

    ix = build_index(str(PDF), chunk_size=size, chunk_overlap=overlap)
    per = []
    for it in ITEMS:
        docs = ix.search(it["question"], k=k, search_type=search_type, translate=query != "ko")
        t0 = time.time()
        out = answer_from_docs(it["question"], docs, style)
        gen_s = time.time() - t0
        v = verify_answer(out["answer"], docs)
        per.append({"id": it["id"], "correct": correct(out["answer"], it["keywords"]), "hit": hit(docs, it["evidence"]),
                    "gen_s": round(gen_s, 1), "input_chars": out["input_chars"], "lines": v["lines"], "bad": v["bad"],
                    "uncited": v["uncited"], "leak": out.get("leak", False), "runaway": out.get("runaway", False), "answer": out["answer"],
                    "reasoning": out.get("reasoning", "")})
        print(f"  {style} {size}/{overlap}/k{k}/{search_type} {it['id']} 정답={per[-1]['correct']} "
              f"{gen_s:.1f}s 걸린줄={v['bad']}/{v['lines']}", flush=True)
    lines = sum(p["lines"] for p in per)
    return {"style": style, "chunk_size": size, "overlap": overlap, "k": k, "search": search_type, "query": query,
            "acc": statistics.mean(p["correct"] for p in per),
            "grounded": 1 - sum(p["bad"] for p in per) / lines if lines else 0.0,
            "uncited": sum(p["uncited"] for p in per),
            "gen_median_s": statistics.median(p["gen_s"] for p in per),
            "input_chars": int(statistics.mean(p["input_chars"] for p in per)),
            "leak": sum(p["leak"] for p in per), "runaway": sum(p["runaway"] for p in per), "items": per}


def report(stage: str, title: str, notes: list[str], results: list[dict]):
    md = [f"# {title}\n", *notes, "",
          "| 프롬프트 | chunk_size/overlap | k | 검색 | 검색 적중 | 정답률 | 근거 일치율 | 출처 없는 줄 | 예시 누출 | 반복 폭주 | 작성 시간(중앙값) | 입력 길이(자) |",
          "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        search = r["search"] + (" · 영어 검색" if r["query"] != "ko" else "")
        hit_rate = statistics.mean(p["hit"] for p in r["items"] if ITEM_BY_ID[p["id"]]["evidence"])
        md.append(f"| {r['style']} | {r['chunk_size']}/{r['overlap']} | {r['k']} | {search} | {hit_rate * 100:.0f}% | {r['acc'] * 100:.0f}% | "
                  f"{r['grounded'] * 100:.0f}% | {r['uncited']} | {r['leak']} | {r['runaway']} | {r['gen_median_s']:.1f}초 | {r['input_chars']:,} |")
    md.append("\n## 문항별 결과\n")
    for r in results:
        md.append(f"### {r['style']} · {r['chunk_size']}/{r['overlap']} · k={r['k']} · {r['search']}\n")
        md.append("| 문항 | 검색 적중 | 정답 | 걸린 줄 | 답변 |\n|---|---|---|---|---|")
        for p in r["items"]:
            ans = squash(p["answer"]).replace("|", "/")[:260]
            md.append(f"| {p['id']} | {'○' if p['hit'] else '×'} | {'○' if p['correct'] else '×'} | {p['bad']}/{p['lines']} | {ans} |")
        md.append("")
    save(stage, "\n".join(md), [{k: v for k, v in r.items()} for r in results])


COMMON_NOTES = [
    "- 문서: `keytruda_label.pdf` (FDA 허가 라벨, 106쪽) · 질문 13개(12개는 문서에 답이 있음, 1개는 없음 → '확인되지 않음'이 정답)",
    "- **정답률**: 정답 키워드(숫자·단위)가 답에 모두 들어 있는 비율",
    "- **근거 일치율**: 출처를 단 줄 중 인용 실재·숫자 일치·AI 검사관 판정을 모두 통과한 비율 (출처 없는 줄은 불합격으로 셈)",
    "- **예시 누출**: Few-shot 예시에만 있는 표현이 실제 답에 섞여 나온 횟수",
    "- **반복 폭주**: 같은 구절을 끝없이 반복해 다시 쓰게 한 횟수 (다시 써도 폭주하면 답 없음 → 오답)",
]


def stage_prompt():
    from bio_gpt.agent.prompts import DOC_STYLES

    # 1단계에서 적중률이 높고 입력이 과하지 않았던 설정으로 고정해, 검색 실패가 프롬프트 비교를 가리지 않게 한다
    results = [run_answers(style, 800, 0, 5, "hybrid") for style in DOC_STYLES]
    report("prompt", "2단계: 프롬프트 기법 비교 (검색 설정 고정: 800/0, k=5, hybrid)",
           COMMON_NOTES, results)


def stage_params(style: str):
    # 기준선(기본 설정) + 1단계 상위 설정
    configs = [(400, 100, 3, "similarity", "ko"), (400, 100, 5, "similarity", "ko"), (800, 0, 5, "hybrid", "ko"),
               (800, 0, 8, "hybrid", "ko"), (400, 0, 8, "hybrid", "en"), (1500, 0, 5, "similarity", "ko")]
    results = [run_answers(style, *c) for c in configs]
    report("params", f"3단계: 검색 파라미터별 실제 답 비교 (프롬프트 고정: {style})", COMMON_NOTES, results)


def stage_final():
    """시작점(기본 설정)과 최종 앱 흐름(검사·고쳐 쓰기·마무리 포함)의 최종 답을 비교한다."""
    from bio_gpt.agent import ask
    from bio_gpt.rag import DEFAULT_STYLE, DEFAULTS

    start = run_answers("baseline", 400, 100, 3, "similarity")
    d = DEFAULTS
    ix = build_index(str(PDF), chunk_size=d["chunk_size"], chunk_overlap=d["chunk_overlap"])
    rows = []
    for it in ITEMS:
        s = ask(it["question"], options={"doc_index": ix, "doc_only": True, "prompt_style": DEFAULT_STYLE,
                                         "rag": {"k": d["k"], "search_type": d["search_type"], "translate": d["translate"]}})
        body = s["final"].split("\n---\n")[0]
        rows.append({"id": it["id"], "status": s["status"], "correct": correct(body, it["keywords"]),
                     "latency": s["latency_ms"] / 1000, "answer": body})
        print(f"  final {it['id']} {s['status']} 정답={rows[-1]['correct']} {rows[-1]['latency']:.1f}s", flush=True)
    status = {k: sum(r["status"] == k for r in rows) for k in ("answered", "partial", "withheld", "no_evidence")}
    md = ["# 4단계: 시작점 vs 최종 앱 (사용자가 실제로 보는 답)\n", *COMMON_NOTES, "",
          "| 구성 | 정답률 | 출처 없는 줄 | 응답 시간(중앙값) |", "|---|---:|---:|---:|",
          f"| 시작점: 기본 설정 (400/100 · k=3 · similarity · 기본 프롬프트, 검증 없음) | {start['acc'] * 100:.0f}% | "
          f"{start['uncited']} | {start['gen_median_s']:.1f}초(작성만) |",
          f"| 최종: {d['chunk_size']}/{d['chunk_overlap']} · k={d['k']} · {d['search_type']} · {DEFAULT_STYLE} + 검사·고쳐 쓰기 | "
          f"{statistics.mean(r['correct'] for r in rows) * 100:.0f}% | 0 | {statistics.median(r['latency'] for r in rows):.1f}초(전체) |",
          "", f"최종 상태: {status}", "", "| 문항 | 상태 | 정답 | 시간 | 최종 답 |", "|---|---|---|---:|---|"]
    md += [f"| {r['id']} | {r['status']} | {'○' if r['correct'] else '×'} | {r['latency']:.1f}초 | "
           f"{squash(r['answer']).replace('|', '/')[:220]} |" for r in rows]
    save("final", "\n".join(md), {"start": start, "final": rows})


def rescore(path: str):
    """저장된 답(json)을 현재 정답 기준으로 다시 채점해 리포트를 다시 만든다 (LLM 호출 없음)."""
    p = Path(path)
    results = json.loads(p.read_text())
    for r in results:
        for it in r["items"]:
            it["correct"] = correct(it["answer"], ITEM_BY_ID[it["id"]]["keywords"])
        r["acc"] = statistics.mean(it["correct"] for it in r["items"])
    title = p.read_text().split("\n")[0] if p.suffix == ".md" else Path(path).with_suffix(".md").read_text().split("\n")[0]
    stage = "prompt" if "rag_prompt" in p.name else "params"
    report(stage, title.lstrip("# ") + " — 재채점", COMMON_NOTES, results)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "retrieval"
    if stage == "retrieval":
        stage_retrieval()
    elif stage == "prompt":
        stage_prompt()
    elif stage == "final":
        stage_final()
    elif stage == "rescore":
        rescore(sys.argv[2])
    elif stage == "params":
        stage_params(sys.argv[2] if len(sys.argv) > 2 else "v1.5")
