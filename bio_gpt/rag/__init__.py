"""업로드 문서(PDF) RAG.

  로드·분할(loader) → 임베딩(embeddings) → FAISS 색인(index) → 검색(index.DocIndex.search)
  검색 방식: similarity / mmr / hybrid(의미 + BM25를 RRF로 합침, retrieval)

검색 결과는 외부 DB와 같은 Doc 형식이라 이후 답변 작성·검사 단계는 그대로 쓴다.
파라미터 비교 실험은 eval/rag_sweep.py, 결과는 docs/rag_experiments.md.
"""
from ..config import EMBED_MODEL
from .index import DEFAULT_STYLE, DEFAULTS, SEARCH_TYPES, DocIndex, build_index

__all__ = ["DEFAULTS", "DEFAULT_STYLE", "SEARCH_TYPES", "DocIndex", "EMBED_MODEL", "build_index"]
