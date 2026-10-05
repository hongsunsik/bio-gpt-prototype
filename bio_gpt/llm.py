"""LLM 호출 (SFR-001, COR-001).

OpenAI 호환 Chat Completions API 하나만 쓴다. Ollama, Gemini, OpenAI 모두 이 형식을 지원해서
config.py의 주소와 모델 이름만 바꾸면 로컬(폐쇄망)과 클라우드를 오갈 수 있다.
"""
import json
import re
import time

from openai import OpenAI, RateLimitError

from .config import JUDGE, WRITER, Endpoint

# 작은 모델이 temperature 0에서 같은 구절을 끝없이 반복하는 경우가 있어 출력 길이를 막아 둔다.
MAX_TOKENS = 1500
RATE_LIMIT_RETRIES = 5

_clients: dict[str, OpenAI] = {}


def _client(ep: Endpoint) -> OpenAI:
    if ep.base_url not in _clients:
        _clients[ep.base_url] = OpenAI(base_url=ep.base_url, api_key=ep.api_key, timeout=180)
    return _clients[ep.base_url]


def _create_with_retry(ep: Endpoint, **kwargs):
    """무료 API의 분당 호출 제한(429)은 기다렸다 다시 부른다. 하루 한도 초과는 기다려도 안 풀리니 바로 올린다."""
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        try:
            return _client(ep).chat.completions.create(model=ep.model, **kwargs)
        except RateLimitError as e:
            if attempt == RATE_LIMIT_RETRIES or "PerDay" in str(e):
                raise
            time.sleep(10 * (attempt + 1))


def complete(messages: list, *, json_mode: bool = False, temperature: float = 0.0, judge: bool = False) -> str:
    ep = JUDGE if judge else WRITER
    kwargs = {"messages": messages, "temperature": temperature, "max_tokens": MAX_TOKENS}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if ep.supports_reasoning_off:
        kwargs["reasoning_effort"] = "none"  # 사고 과정 출력을 끄면 빠르고 JSON도 덜 깨진다
    resp = _create_with_retry(ep, **kwargs)
    text = resp.choices[0].message.content or ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def chat(system: str, user: str, *, json_mode: bool = False, temperature: float = 0.0) -> str:
    return complete([{"role": "system", "content": system}, {"role": "user", "content": user}],
                    json_mode=json_mode, temperature=temperature)


def _parse_json(text: str) -> dict | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)  # 앞뒤에 설명이 붙은 경우
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def chat_json(system: str, user: str, messages: list | None = None, judge: bool = False) -> dict:
    """JSON 응답을 받는다. 파싱에 실패하면 한 번 더 부른다. messages를 주면 그 대화에 이어서 부른다."""
    msgs = messages or [{"role": "system", "content": system}, {"role": "user", "content": user}]
    text = ""
    for _ in range(2):
        text = complete(msgs, json_mode=True, judge=judge)
        parsed = _parse_json(text)
        if parsed is not None:
            return parsed
    raise ValueError(f"LLM이 올바른 JSON을 반환하지 않음: {text[:200]}")
