"""LLM 연동 계층 (SFR-001 LLM 모델 선정 / COR-001 표준 API).

OpenAI 호환 Chat Completions API 하나로 통일해, 환경변수만 바꾸면
로컬(Ollama, 폐쇄망) ↔ 클라우드(GPT-4o, Gemini) 모델을 교체할 수 있다.

  # 로컬 (기본값)
  LLM_BASE_URL=http://localhost:11434/v1  LLM_MODEL=bio-qwen3
  # OpenAI
  LLM_BASE_URL=https://api.openai.com/v1  LLM_MODEL=gpt-4o  LLM_API_KEY=sk-...
  # Gemini (OpenAI 호환 엔드포인트)
  LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/  LLM_MODEL=gemini-3.8-flash  LLM_API_KEY=...

간단히 LLM_PROFILE=local|gemini|openai 로 고를 수도 있다 (키는 GEMINI_API_KEY / OPENAI_API_KEY).
"""
import json
import os
import re
import time

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

load_dotenv()

PROFILES = {
    "local": ("http://localhost:11434/v1", "bio-qwen3", "ollama"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-3.8-flash", os.getenv("GEMINI_API_KEY", "")),
    "openai": ("https://api.openai.com/v1", "gpt-4o", os.getenv("OPENAI_API_KEY", "")),
}
PROFILE = os.getenv("LLM_PROFILE", "local")
_base, _model, _key = PROFILES[PROFILE]
BASE_URL = os.getenv("LLM_BASE_URL", _base)
MODEL = os.getenv("LLM_MODEL", _model)
API_KEY = os.getenv("LLM_API_KEY", _key)
IS_LOCAL = "11434" in BASE_URL
IS_GEMINI = "generativelanguage" in BASE_URL

_client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=180)

# 검사관(답변 검사)만 다른 모델로 돌릴 수 있다. 예: JUDGE_PROFILE=gemini
# 작성 모델이 자기 답을 너그럽게 채점하는 위험을 줄이고, 작은 로컬 모델의 오탐을 줄이기 위함 (v1.5)
JUDGE_PROFILE = os.getenv("JUDGE_PROFILE", "")
if JUDGE_PROFILE:
    _jb, _jm, _jk = PROFILES[JUDGE_PROFILE]
    JUDGE_MODEL = os.getenv("JUDGE_MODEL", _jm)
    _judge_client = OpenAI(base_url=_jb, api_key=_jk, timeout=180)
    _JUDGE_LOCAL_OR_GEMINI = "11434" in _jb or "generativelanguage" in _jb
else:
    JUDGE_MODEL, _judge_client, _JUDGE_LOCAL_OR_GEMINI = MODEL, _client, IS_LOCAL or IS_GEMINI


def complete(messages: list, *, json_mode: bool = False, temperature: float = 0.0, judge: bool = False) -> str:
    client, model = (_judge_client, JUDGE_MODEL) if judge else (_client, MODEL)
    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if (_JUDGE_LOCAL_OR_GEMINI if judge else IS_LOCAL or IS_GEMINI):
        # 사고(thinking) 출력을 끈다. 지연 시간을 줄이고 JSON 파싱을 안정화. (Gemini Flash도 지원)
        kwargs["reasoning_effort"] = "none"
    for attempt in range(6):
        try:
            resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature, **kwargs)
            break
        except RateLimitError as e:  # 무료 API 분당 호출 제한 → 대기 후 재시도
            if attempt == 5 or "PerDay" in str(e):  # 하루 한도 초과는 기다려도 소용없으므로 바로 중단
                raise
            time.sleep(10 * (attempt + 1))
    text = resp.choices[0].message.content or ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def chat(system: str, user: str, *, json_mode: bool = False, temperature: float = 0.0) -> str:
    return complete([{"role": "system", "content": system}, {"role": "user", "content": user}],
                    json_mode=json_mode, temperature=temperature)


def chat_json(system: str, user: str, messages: list | None = None, judge: bool = False) -> dict:
    """JSON 응답을 강제하고, 파싱 실패 시 한 번 재시도한다. messages를 주면 그 대화를 이어서 호출."""
    msgs = messages or [{"role": "system", "content": system}, {"role": "user", "content": user}]
    last = ""
    for _ in range(2):
        last = complete(msgs, json_mode=True, judge=judge)
        try:
            return json.loads(last)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", last, flags=re.S)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass
    raise ValueError(f"LLM이 올바른 JSON을 반환하지 않음: {last[:200]}")
