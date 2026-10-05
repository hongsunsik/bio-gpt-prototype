"""환경변수로 바꾸는 설정을 한곳에 모은다 (COR-001).

모델 교체, 검사관 분리, 임베딩 선택은 모두 .env만 고치면 되고 코드는 건드리지 않는다.

  LLM_PROFILE=local|gemini|openai     작성 모델 (기본: OPENAI_API_KEY가 있으면 openai, 없으면 local)
  LLM_BASE_URL / LLM_MODEL / LLM_API_KEY   프로필 값을 하나씩 덮어쓸 때
  JUDGE_PROFILE=gemini                검사관만 다른 모델로 (비우면 작성 모델과 같음)
  EMBED_PROFILE=local|openai          업로드 문서 임베딩 (기본: LLM_PROFILE을 따라감)
"""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"  # .gitignore 대상
OLLAMA_URL = "http://localhost:11434/v1"


@dataclass(frozen=True)
class Endpoint:
    base_url: str
    model: str
    api_key: str

    @property
    def is_local(self) -> bool:
        return "11434" in self.base_url

    @property
    def is_gemini(self) -> bool:
        return "generativelanguage" in self.base_url

    @property
    def supports_reasoning_off(self) -> bool:
        """qwen3(Ollama)와 Gemini Flash는 reasoning_effort="none"으로 사고 출력을 끌 수 있다."""
        return self.is_local or self.is_gemini


PROFILES = {
    "local": Endpoint(OLLAMA_URL, "bio-qwen3", "ollama"),
    "gemini": Endpoint("https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-3.8-flash",
                       os.getenv("GEMINI_API_KEY", "")),
    "openai": Endpoint("https://api.openai.com/v1", "gpt-4o", os.getenv("OPENAI_API_KEY", "")),
}

PROFILE = os.getenv("LLM_PROFILE") or ("openai" if os.getenv("OPENAI_API_KEY") else "local")
WRITER = Endpoint(
    base_url=os.getenv("LLM_BASE_URL", PROFILES[PROFILE].base_url),
    model=os.getenv("LLM_MODEL", PROFILES[PROFILE].model),
    api_key=os.getenv("LLM_API_KEY", PROFILES[PROFILE].api_key),
)

JUDGE_PROFILE = os.getenv("JUDGE_PROFILE", "")
if JUDGE_PROFILE:
    _j = PROFILES[JUDGE_PROFILE]
    JUDGE = Endpoint(_j.base_url, os.getenv("JUDGE_MODEL", _j.model), _j.api_key)
else:
    JUDGE = WRITER

EMBED_PROFILE = os.getenv("EMBED_PROFILE", "openai" if PROFILE == "openai" else "local")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small" if EMBED_PROFILE == "openai" else "bge-m3")
