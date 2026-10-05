"""여러 계층이 같이 쓰는 데이터 타입."""
from dataclasses import asdict, dataclass, field


@dataclass
class Doc:
    """검색된 근거 문서 한 건. 출처와 상관없이 같은 모양으로 다뤄야 인용 검사를 한 번에 할 수 있다.

    id는 답변에 그대로 적는 인용 키다.
      PMID:12345678 (논문) / NCT01234567 (임상) / FDA:0098dec4-IND (허가 라벨 섹션) / DOC:p12-3 (업로드 문서 청크)
    """
    id: str
    source: str  # pubmed | trials | fda | pdf
    title: str
    url: str
    text: str
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
