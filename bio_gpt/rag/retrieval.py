"""검색 보조: 키워드 검색(BM25), 순위 합치기(RRF), 질문 번역."""
import re

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from ..llm import complete

RRF_K = 60  # RRF 원 논문(Cormack et al., 2009)의 기본값


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9가-힣][a-z0-9가-힣\-./%]*", text.lower())


class KeywordIndex:
    """BM25: 질문 단어가 청크에 얼마나 자주, 그리고 문서 전체에서 얼마나 드물게 나오는지로 점수를 매긴다."""

    def __init__(self, docs: list[Document]):
        self.docs = docs
        self.bm25 = BM25Okapi([tokenize(d.page_content) for d in docs])

    def search(self, query: str, n: int) -> list[Document]:
        scores = self.bm25.get_scores(tokenize(query))
        top = sorted(range(len(self.docs)), key=lambda i: -scores[i])[:n]
        return [self.docs[i] for i in top if scores[i] > 0]


def rrf_fuse(rankings: list[list[Document]], k: int, key=lambda d: d.metadata["cite_id"]) -> list[Document]:
    """Reciprocal Rank Fusion. 순위 r에 1/(60+r)점을 주고 더한다.
    의미 검색 거리와 BM25 점수는 단위가 달라 바로 더할 수 없지만 순위끼리는 합칠 수 있다."""
    scored: dict[str, list] = {}
    for docs in rankings:
        for rank, d in enumerate(docs):
            entry = scored.setdefault(key(d), [d, 0.0])
            entry[1] += 1 / (RRF_K + rank)
    return [d for d, _ in sorted(scored.values(), key=lambda e: -e[1])[:k]]


_translations: dict[str, str] = {}

TRANSLATE_SYSTEM = ("Rewrite the Korean question as a short English search query for a drug label. "
                    "Use standard medical terms and keep drug names, trial names and numbers. Output only the query.")


def translate_query(question: str) -> str:
    """한국어 질문을 영어 검색어로 바꾼다 (영어 문서를 키워드로 찾을 때). 같은 질문은 캐시에서 꺼낸다."""
    if not re.search(r"[가-힣]", question):
        return question
    if question not in _translations:
        out = complete([{"role": "system", "content": TRANSLATE_SYSTEM}, {"role": "user", "content": question}])
        _translations[question] = out.strip().strip('"')
    return _translations[question]
