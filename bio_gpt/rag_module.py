"""업로드 문서(PDF) RAG 모듈 — 2주차 실습 가이드의 8단계 파이프라인.

  1. 문서 로드     PyMuPDFLoader            PDF → 페이지별 텍스트
  2. 분할          RecursiveCharacterTextSplitter (chunk_size, chunk_overlap)
  3. 임베딩        bge-m3(로컬, 한·영 다국어) 또는 OpenAI text-embedding-3-small
  4. 벡터 DB 저장  FAISS
  5. 검색기        similarity(유사도 상위 k개) · mmr(비슷한 청크끼리 중복 줄이기)
                   · hybrid(의미 검색 + 키워드 검색 BM25를 순위로 합침) — 실험으로 추가
                   translate=True면 질문을 LLM으로 영어 검색어로 바꿔 검색한다(문서가 영어일 때)
  6~8. 프롬프트 · LLM · 출력은 기존 Bio-GPT 흐름(graph.py)을 그대로 쓴다.
       → 청크마다 출처 ID(DOC:p<쪽>-<번호>)를 붙여서 '모든 문장에 출처 + 자동 검증'이 업로드 문서에도 적용된다.

파라미터를 바꾸며 결과를 비교하는 실험은 eval/rag_sweep.py.
"""
import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rank_bm25 import BM25Okapi

from .llm import PROFILE, complete
from .sources import Doc

# 가이드 예제 코드의 값은 400 / 100 / k=3 / similarity.
# 실험(eval/rag_sweep.py, docs/rag_experiments.md) 결과 정답률이 가장 높았던 값으로 바꿨다.
DEFAULTS = {"chunk_size": 800, "chunk_overlap": 0, "k": 5, "search_type": "hybrid", "translate": False}
DEFAULT_STYLE = "fewshot"

# 임베딩: OpenAI 프로필이면 OpenAI 임베딩, 아니면 로컬 Ollama의 bge-m3.
# bge-m3는 다국어 모델이라 한국어 질문으로 영어 문서를 바로 찾을 수 있다.
if os.getenv("EMBED_PROFILE", "openai" if PROFILE == "openai" else "local") == "openai":
    EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    _embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
else:
    EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
    # Ollama의 OpenAI 호환 주소. 토큰 길이 검사(tiktoken)는 OpenAI 모델 전용이라 끈다.
    # 청크 1000개를 한 번에 보내면 Ollama가 도중에 연결을 끊어서(실측) 50개씩 나눠 보낸다.
    _embeddings = OpenAIEmbeddings(model=EMBED_MODEL, base_url="http://localhost:11434/v1", api_key="ollama",
                                   check_embedding_ctx_length=False, chunk_size=50)


@dataclass
class DocIndex:
    """PDF 하나를 분할·임베딩해 둔 검색용 색인."""
    name: str
    pages: int
    chunks: int
    chunk_size: int
    chunk_overlap: int
    store: FAISS
    summary: str = ""  # 첫 쪽 앞부분
    _bm25: tuple | None = field(default=None, repr=False)

    @property
    def text(self) -> str:
        """문서 전체 글(소문자). 질문 속 약물 이름이 이 문서에 나오는지 확인할 때 쓴다."""
        if not hasattr(self, "_text"):
            self._text = re.sub(r"\s+", " ", " ".join(d.page_content for d in self.store.docstore._dict.values())).lower()
        return self._text

    def _keyword_search(self, query: str, n: int) -> list:
        """BM25 키워드 검색: 질문의 단어가 청크에 몇 번, 얼마나 드물게 나오는지로 점수를 매긴다."""
        if self._bm25 is None:
            docs = list(self.store.docstore._dict.values())
            self._bm25 = (BM25Okapi([_tokens(d.page_content) for d in docs]), docs)
        bm25, docs = self._bm25
        scores = bm25.get_scores(_tokens(query))
        top = sorted(range(len(docs)), key=lambda i: -scores[i])[:n]
        return [docs[i] for i in top if scores[i] > 0]

    def search(self, query: str, k: int = 3, search_type: str = "similarity", translate: bool = False) -> list[Doc]:
        """[5단계] 질문과 가까운 청크 k개를 Bio-GPT 문서(Doc) 형식으로 돌려준다."""
        if translate:
            query = translate_query(query)
        if search_type == "mmr":
            # 후보를 넉넉히(k×4) 뽑은 뒤, 서로 비슷한 청크는 빼고 k개를 고른다
            found = [(d, None) for d in self.store.max_marginal_relevance_search(query, k=k, fetch_k=k * 4)]
        elif search_type == "hybrid":
            # 두 검색의 순위를 RRF(1/(60+순위)의 합)로 합친다. 점수 단위가 달라도 순위끼리는 합칠 수 있다.
            n = max(k * 4, 20)
            ranked: dict[str, list] = {}
            for docs in ([d for d, _ in self.store.similarity_search_with_score(query, k=n)], self._keyword_search(query, n)):
                for rank, d in enumerate(docs):
                    entry = ranked.setdefault(d.metadata["cite_id"], [d, 0.0])
                    entry[1] += 1 / (60 + rank)
            found = [(d, None) for d, _ in sorted(ranked.values(), key=lambda e: -e[1])[:k]]
        else:
            found = self.store.similarity_search_with_score(query, k=k)  # 점수 = 거리(작을수록 가까움)
        return [Doc(id=d.metadata["cite_id"], source="pdf", title=f"{self.name} {d.metadata['page'] + 1}쪽",
                    url="", text=d.page_content,
                    meta={"page": d.metadata["page"] + 1, "distance": None if s is None else round(float(s), 3)})
                for d, s in found]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9가-힣][a-z0-9가-힣\-./%]*", text.lower())


_translations: dict[str, str] = {}


def translate_query(question: str) -> str:
    """한국어 질문 → 영어 검색어 (영어 문서를 키워드로 찾기 위해). 같은 질문은 다시 부르지 않는다."""
    if not re.search(r"[가-힣]", question):
        return question
    if question not in _translations:
        _translations[question] = complete([
            {"role": "system", "content": "Rewrite the Korean question as a short English search query for a drug label. "
                                          "Use standard medical terms and keep drug names, trial names and numbers. "
                                          "Output only the query."},
            {"role": "user", "content": question}]).strip().strip('"')
    return _translations[question]


def load_and_split(pdf_path: str, chunk_size: int, chunk_overlap: int) -> tuple[list, int]:
    """[1~2단계] PDF를 읽고 청크로 나눈다. 청크마다 인용용 ID를 붙인다."""
    pages = PyMuPDFLoader(pdf_path).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    per_page: dict[int, int] = {}
    for c in chunks:
        p = c.metadata["page"]
        per_page[p] = per_page.get(p, 0) + 1
        c.metadata["cite_id"] = f"DOC:p{p + 1}-{per_page[p]}"  # 예: DOC:p12-3 = 12쪽의 3번째 청크
    return chunks, len(pages)


_cache: dict[tuple, DocIndex] = {}
INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "index"  # .gitignore 대상


def build_index(pdf_path: str, name: str | None = None, chunk_size: int = DEFAULTS["chunk_size"],
                chunk_overlap: int = DEFAULTS["chunk_overlap"]) -> DocIndex:
    """[1~4단계] 같은 파일·같은 분할 설정이면 만들어 둔 색인을 다시 쓴다(임베딩 비용 절약)."""
    with open(pdf_path, "rb") as f:
        digest = hashlib.md5(f.read()).hexdigest()
    key = (digest, chunk_size, chunk_overlap, EMBED_MODEL)
    if key not in _cache:
        # 디스크에 저장해 둔 색인이 있으면 불러온다 (앱을 다시 켜도 임베딩을 다시 안 함 → 데모 대기 시간 0)
        folder = INDEX_DIR / f"{digest[:12]}_{chunk_size}_{chunk_overlap}_{EMBED_MODEL.replace('/', '-')}"
        if folder.exists():
            store = FAISS.load_local(str(folder), _embeddings, allow_dangerous_deserialization=True)  # 내가 만든 파일만 읽음
            n_pages = max(d.metadata["page"] for d in store.docstore._dict.values()) + 1
            n_chunks = len(store.docstore._dict)
        else:
            chunks, n_pages = load_and_split(pdf_path, chunk_size, chunk_overlap)
            store = FAISS.from_documents(chunks, _embeddings)  # [3~4단계] 임베딩 + FAISS 저장
            store.save_local(str(folder))
            n_chunks = len(chunks)
        first = min(store.docstore._dict.values(), key=lambda d: (d.metadata["page"], d.metadata["cite_id"]))
        summary = re.sub(r"\s+", " ", first.page_content)[:300]
        _cache[key] = DocIndex(name or os.path.basename(pdf_path), n_pages, n_chunks, chunk_size, chunk_overlap, store, summary)
    return _cache[key]
