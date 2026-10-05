"""외부 API 공통 HTTP 호출 (SIR-001 재시도 정책)."""
import time

import httpx

TIMEOUT = 15
HEADERS = {"User-Agent": "bio-gpt-prototype/0.1 (education project)"}
RETRYABLE = {429}


def get(url: str, params: dict, retries: int = 2) -> httpx.Response:
    """타임아웃·5xx·429는 0.8초, 1.6초 간격으로 다시 시도한다. 404는 '결과 없음'이라 그대로 돌려준다."""
    for attempt in range(retries + 1):
        try:
            r = httpx.get(url, params=params, timeout=TIMEOUT, headers=HEADERS)
            if r.status_code == 404:
                return r
            if r.status_code in RETRYABLE or r.status_code >= 500:
                raise httpx.HTTPStatusError("retryable", request=r.request, response=r)
            r.raise_for_status()
            return r
        except (httpx.TransportError, httpx.HTTPStatusError):
            if attempt == retries:
                raise
            time.sleep(0.8 * (2**attempt))
    raise RuntimeError("unreachable")
