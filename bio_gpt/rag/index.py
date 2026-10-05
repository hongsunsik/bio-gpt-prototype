"""PDF 한 개의 검색 색인. 만든 색인은 메모리와 디스크(data/index/)에 저장해 다시 쓴다."""
import hashlib
import os
import re
from dataclasses import dataclass, field

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from ..config import DATA_DIR, EMBED_MODEL
from ..models import Doc
from .embeddings import make_embeddings
from .loader import load_and_split
from .retrieval import KeywordIndex, rrf_fuse, translate_query

# 실습 예제 값은 400 / 100 / k=3 / similarity였다. 실험(docs/rag_experiments.md)에서 정답률이 가장 높았던 값.
DEFAULTS = {"chunk_size": 800, "chunk_overlap": 0, "k": 5, "search_type": "hybrid", "translate": False}
DEFAULT_STYLE = "fewshot"
SEARCH_TYPES = ("similarity", "mmr", "hybrid")
INDEX_DIR = DATA_DIR / "index"

_embeddings = make_embeddings()


@dataclass
class DocIndex:
    name: str
    pages: int
    chunks: int
    chunk_size: int
    chunk_overlap: int
    store: FAISS
    summary: str = ""  # 첫 쪽 앞부분
    _keyword: KeywordIndex | None = field(default=None, repr=False)
    _text: str | None = field(default=None, repr=False)

    @property
    def documents(self) -> list[Document]:
        return list(self.store.docstore._dict.values())

    @property
    def text(self) -> str:
        """문서 전체(소문자). 질문 속 약물 이름이 이 문서에 나오는지 볼 때 쓴다."""
        if self._text is None:
            self._text = re.sub(r"\s+", " ", " ".join(d.page_content for d in self.documents)).lower()
        return self._text

    def _keyword_search(self, query: str, n: int) -> list[Document]:
        if self._keyword is None:
            self._keyword = KeywordIndex(self.documents)
        return self._keyword.search(query, n)

    def _retrieve(self, query: str, k: int, search_type: str) -> list[tuple[Document, float | None]]:
        if search_type == "mmr":
            # 후보를 k×4개 뽑고, 서로 비슷한 청크는 빼면서 k개를 고른다
            return [(d, None) for d in self.store.max_marginal_relevance_search(query, k=k, fetch_k=k * 4)]
        if search_type == "hybrid":
            n = max(k * 4, 20)
            semantic = [d for d, _ in self.store.similarity_search_with_score(query, k=n)]
            return [(d, None) for d in rrf_fuse([semantic, self._keyword_search(query, n)], k)]
        return self.store.similarity_search_with_score(query, k=k)  # 점수 = L2 거리(작을수록 가까움)

    def search(self, query: str, k: int = 3, search_type: str = "similarity", translate: bool = False) -> list[Doc]:
        if translate:
            query = translate_query(query)
        return [Doc(id=d.metadata["cite_id"], source="pdf", title=f"{self.name} {d.metadata['page'] + 1}쪽",
                    url="", text=d.page_content,
                    meta={"page": d.metadata["page"] + 1, "distance": None if s is None else round(float(s), 3)})
                for d, s in self._retrieve(query, k, search_type)]


_cache: dict[tuple, DocIndex] = {}


def _file_digest(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def _load_or_build(pdf_path: str, folder, chunk_size: int, chunk_overlap: int) -> tuple[FAISS, int, int]:
    if folder.exists():
        # 직접 만든 파일만 읽으므로 pickle 역직렬화를 허용한다
        store = FAISS.load_local(str(folder), _embeddings, allow_dangerous_deserialization=True)
        docs = store.docstore._dict.values()
        return store, max(d.metadata["page"] for d in docs) + 1, len(docs)
    chunks, n_pages = load_and_split(pdf_path, chunk_size, chunk_overlap)
    store = FAISS.from_documents(chunks, _embeddings)
    store.save_local(str(folder))
    return store, n_pages, len(chunks)


def build_index(pdf_path: str, name: str | None = None, chunk_size: int = DEFAULTS["chunk_size"],
                chunk_overlap: int = DEFAULTS["chunk_overlap"]) -> DocIndex:
    """파일 내용과 분할 설정이 같으면 만들어 둔 색인을 다시 쓴다 (임베딩은 비싸다)."""
    digest = _file_digest(pdf_path)
    key = (digest, chunk_size, chunk_overlap, EMBED_MODEL)
    if key not in _cache:
        folder = INDEX_DIR / f"{digest[:12]}_{chunk_size}_{chunk_overlap}_{EMBED_MODEL.replace('/', '-')}"
        store, n_pages, n_chunks = _load_or_build(pdf_path, folder, chunk_size, chunk_overlap)
        first = min(store.docstore._dict.values(), key=lambda d: (d.metadata["page"], d.metadata["cite_id"]))
        summary = re.sub(r"\s+", " ", first.page_content)[:300]
        _cache[key] = DocIndex(name or os.path.basename(pdf_path), n_pages, n_chunks, chunk_size, chunk_overlap,
                               store, summary)
    return _cache[key]
