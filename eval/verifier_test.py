"""검증기 자체 평가: 일부러 틀린 문장을 섞은 답변을 검증기가 잡아내는지 측정한다.

검증기가 너무 관대하면 환각이 통과하고, 너무 엄격하면 정상 답변이 막힌다(1차 평가의 오탐 사례).
그래서 정답(참/거짓)을 아는 문장으로 검증기의 탐지율·오탐률을 따로 잰다.

실행: uv run python eval/verifier_test.py
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import bio_gpt.graph as G  # noqa: E402
from bio_gpt import prompts as P  # noqa: E402
from bio_gpt.sources import search_fda_label, search_pubmed  # noqa: E402

# (자료 가져오기, 질문, [(문장, 참 여부)])
def fda(drug, section):
    return lambda: search_fda_label(drug, [section])


def pubmed(query, pmid):
    return lambda: [d for d in search_pubmed(query) if d.id == pmid]


CASES = [
    (fda("metformin", "DOSE"), "메트포르민의 FDA 라벨상 권장 용법·용량은?", [
        ("성인 시작 용량은 500 mg 하루 2회 또는 850 mg 하루 1회이며 식사와 함께 복용한다", True),
        ("성인 최대 권장 용량은 하루 2550 mg이다", True),
        ("소아 환자에게는 사용이 금지되어 있다", False),
        ("공복에 복용하는 것이 권장된다", False),
        ("신장애 환자에게도 신기능 평가 없이 사용할 수 있다", False),
    ]),
    (fda("trastuzumab deruxtecan", "BOX"), "트라스투주맙 데룩스테칸의 박스 경고 내용은?", [
        ("간질성 폐질환과 폐장염으로 사망한 사례가 보고되었다", True),
        ("Grade 2 이상의 간질성 폐질환이 발생하면 영구 중단한다", True),
        ("박스 경고에는 심장 독성(좌심실 기능 저하)이 포함된다", False),
        ("임신 중 투여는 태아에게 안전한 것으로 명시되어 있다", False),
        ("간질성 폐질환은 경미한 사례만 보고되었다", False),
    ]),
    (fda("lecanemab", "IND"), "레카네맙의 FDA 허가 적응증은?", [
        ("알츠하이머병 치료에 허가되었다", True),
        ("경도인지장애 또는 경도 치매 단계 환자에게 투여를 시작한다", True),
        ("중증 치매 환자에게도 투여를 시작할 수 있다", False),
        ("파킨슨병 치료에도 허가되었다", False),
    ]),
    # 한국어 번역 표현(3상, 이중눈가림 등)을 원문과 같은 뜻으로 알아보는지 (v1.2 오탐 사례)
    (pubmed("lecanemab Alzheimer's disease phase 3", "PMID:36449413"), "레카네맙 3상 임상 결과는?", [
        ("3상 임상시험으로 진행되었다", True),
        ("18개월 동안 진행된 이중눈가림 시험이다", True),
        ("50~90세의 초기 알츠하이머병 환자를 대상으로 했다", True),
        ("공개 라벨(비맹검) 시험이었다", False),
        ("중증 치매 환자를 대상으로 했다", False),
    ]),
    # 같은 뜻을 다른 말로 쓴 문장을 통과시키는지 (v1.3 오탐 사례: '수술 후 보조 치료' vs '보조요법')
    (fda("pembrolizumab", "IND"), "펨브롤리주맙의 FDA 허가 적응증은?", [
        ("흑색종의 수술 후 보조 치료에 허가되었다", True),
        ("완전 절제 후 흑색종 환자의 보조요법으로 쓸 수 있다", True),
        ("비소세포폐암에도 허가되어 있다", True),
        ("흑색종에는 허가되지 않았다", False),
        ("알츠하이머병 치료에 허가되었다", False),
        # 문서 일부에 안 나온다고 '없다'고 단정한 문장 (v1.4에서 실제로 통과된 오류 유형)
        ("자궁내막암에는 허가되지 않았다", False),
    ]),
]


def run(mode: str) -> dict:
    G.VERIFY_MODE = mode
    tp = fn = fp = tn = 0
    secs = []
    for fetch, question, claims in CASES:
        docs = fetch()
        doc_id = docs[0].id
        answer = "\n".join(f"- {text} [{doc_id}]" for text, _ in claims)
        messages = [{"role": "system", "content": P.GENERATOR_SYSTEM},
                    {"role": "user", "content": P.GENERATOR_USER.format(
                        question=question, glossary="(해당 없음)", context=G._context(docs),
                        ids=", ".join(d.id for d in docs), feedback="")}]
        if mode == "continue":
            G.complete(messages)  # 실제 흐름처럼 생성 호출을 먼저 해서 캐시를 채운다
        state = {"docs": docs, "answer": answer, "gen_messages": messages, "verify_runs": [], "trace": []}
        t0 = time.time()
        rows = G.verify(state)["verification"]["lines"]
        secs.append(time.time() - t0)
        for row, (_, truth) in zip(rows, claims):
            flagged = not G._line_ok(row)
            if truth:
                fp += flagged
                tn += not flagged
            else:
                tp += flagged
                fn += not flagged
    return {"탐지율(거짓 문장 적발)": tp / (tp + fn), "오탐률(참 문장을 거짓으로)": fp / (fp + tn),
            "검증 평균 시간": sum(secs) / len(secs)}


if __name__ == "__main__":
    modes = sys.argv[1:] or ["separate", "continue"]
    print(f"모델: {os.getenv('LLM_PROFILE', 'local')} / 검사관: {os.getenv('JUDGE_PROFILE') or '같은 모델'} {os.getenv('JUDGE_MODEL', '')}")
    for m in modes:
        r = run(m)
        print(f"[{m}] 탐지율 {r['탐지율(거짓 문장 적발)'] * 100:.0f}% · 오탐률 {r['오탐률(참 문장을 거짓으로)'] * 100:.0f}% · "
              f"검증 평균 {r['검증 평균 시간']:.1f}초")
