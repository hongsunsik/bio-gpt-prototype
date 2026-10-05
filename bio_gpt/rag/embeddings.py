"""임베딩 모델 선택. OpenAI 프로필이면 text-embedding-3-small, 아니면 로컬 Ollama의 bge-m3.

bge-m3는 다국어 모델이라 한국어 질문으로 영어 문서를 바로 찾을 수 있다.
"""
from langchain_openai import OpenAIEmbeddings

from ..config import EMBED_MODEL, EMBED_PROFILE, OLLAMA_URL


def make_embeddings() -> OpenAIEmbeddings:
    if EMBED_PROFILE == "openai":
        return OpenAIEmbeddings(model=EMBED_MODEL)
    # tiktoken 길이 검사는 OpenAI 모델 전용이라 끈다.
    # 한 번에 1000개를 보내면 Ollama가 연결을 끊어서 50개씩 나눠 보낸다.
    return OpenAIEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL, api_key="ollama",
                            check_embedding_ctx_length=False, chunk_size=50)
