"""외부 바이오 DB 연동 (SFR-004).

1차년도 데이터 소스는 3곳으로 한정했다. 모든 문서는 출처 ID를 인용 키로 가지므로
답변의 인용이 실제로 검색된 문서인지 코드로 확인할 수 있다 (SFR-005, SFR-006).

새 소스를 붙일 때는 Doc 리스트를 돌려주는 함수를 만들고 SOURCES에 한 줄 추가하면 된다.
"""
from .fda import search_fda_label
from .pubmed import search_pubmed
from .trials import search_trials

SOURCES = {
    "pubmed": search_pubmed,
    "trials": search_trials,
    "fda": search_fda_label,
}

__all__ = ["SOURCES", "search_fda_label", "search_pubmed", "search_trials"]
