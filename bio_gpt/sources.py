"""외부 바이오 DB 연동 (SFR-004, SIR-001).

1차년도 데이터 소스를 피드백대로 3곳으로 한정한다.
  - 논문: PubMed (NCBI E-utilities)
  - 임상: ClinicalTrials.gov API v2
  - 인허가: openFDA Drug Label (미국 FDA 허가 라벨)

모든 문서는 출처 ID(PMID / NCT 번호 / FDA set_id)를 인용 키로 갖는다.
그래서 답변의 인용이 실제 검색된 문서인지 기계적으로 검증할 수 있다(SFR-005, SFR-006).
"""
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field

import httpx

TIMEOUT = 15
UA = {"User-Agent": "bio-gpt-prototype/0.1 (education project)"}


@dataclass
class Doc:
    id: str  # 인용 키 (예: PMID:12345678, NCT01234567, FDA:0098dec4-IND)
    source: str  # pubmed | trials | fda
    title: str
    url: str
    text: str
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _get(url: str, params: dict, retries: int = 2) -> httpx.Response:
    """SIR-001 재시도 정책: 타임아웃·5xx·429에 대해 지수 백오프로 최대 2회 재시도."""
    for attempt in range(retries + 1):
        try:
            r = httpx.get(url, params=params, timeout=TIMEOUT, headers=UA)
            if r.status_code == 404:
                return r
            if r.status_code in (429,) or r.status_code >= 500:
                raise httpx.HTTPStatusError("retryable", request=r.request, response=r)
            r.raise_for_status()
            return r
        except (httpx.TransportError, httpx.HTTPStatusError):
            if attempt == retries:
                raise
            time.sleep(0.8 * (2**attempt))
    raise RuntimeError("unreachable")


# ---------------------------------------------------------------- PubMed
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def search_pubmed(query: str, n: int = 4) -> list[Doc]:
    ids = _get(f"{EUTILS}/esearch.fcgi",
               {"db": "pubmed", "term": query, "retmax": n, "sort": "relevance", "retmode": "json"}
               ).json()["esearchresult"]["idlist"]
    if not ids:
        return []
    xml = _get(f"{EUTILS}/efetch.fcgi", {"db": "pubmed", "id": ",".join(ids), "retmode": "xml"}).text
    docs = []
    for art in ET.fromstring(xml).findall(".//PubmedArticle"):
        pmid = art.findtext(".//PMID")
        title = "".join(art.find(".//ArticleTitle").itertext()) if art.find(".//ArticleTitle") is not None else ""
        abstract = " ".join("".join(a.itertext()) for a in art.findall(".//Abstract/AbstractText"))
        year = art.findtext(".//PubDate/Year") or art.findtext(".//PubDate/MedlineDate") or ""
        journal = art.findtext(".//Journal/Title") or ""
        if not abstract:
            continue
        docs.append(Doc(
            id=f"PMID:{pmid}", source="pubmed", title=title,
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            text=f"{title} ({journal}, {year}). {abstract}"[:1500],
            meta={"year": year, "journal": journal},
        ))
    return docs


# ---------------------------------------------------------------- ClinicalTrials.gov
CTGOV = "https://clinicaltrials.gov/api/v2/studies"


def search_trials(query: str, n: int = 4) -> list[Doc]:
    data = _get(CTGOV, {"query.term": query, "pageSize": n, "format": "json"}).json()
    docs = []
    for s in data.get("studies", []):
        p = s.get("protocolSection", {})
        ident = p.get("identificationModule", {})
        status = p.get("statusModule", {})
        design = p.get("designModule", {})
        nct = ident.get("nctId", "")
        conds = ", ".join(p.get("conditionsModule", {}).get("conditions", []))
        intervs = ", ".join(i.get("name", "") for i in p.get("armsInterventionsModule", {}).get("interventions", []))
        phases = ", ".join(design.get("phases", []) or ["N/A"])
        enroll = design.get("enrollmentInfo", {}).get("count", "")
        summary = p.get("descriptionModule", {}).get("briefSummary", "")
        text = (f"{ident.get('briefTitle', '')}. Status: {status.get('overallStatus', '')}. Phase: {phases}. "
                f"Conditions: {conds}. Interventions: {intervs}. Enrollment: {enroll}. "
                f"Start: {status.get('startDateStruct', {}).get('date', '')}. Summary: {summary}")
        docs.append(Doc(
            id=nct, source="trials", title=ident.get("briefTitle", ""),
            url=f"https://clinicaltrials.gov/study/{nct}", text=text[:1000],
            meta={"status": status.get("overallStatus", ""), "phase": phases},
        ))
    return docs


# ---------------------------------------------------------------- openFDA Drug Label
OPENFDA = "https://api.fda.gov/drug/label.json"
LABEL_SECTIONS = {"IND": "indications_and_usage", "DOSE": "dosage_and_administration", "BOX": "boxed_warning"}
SECTION_BUDGET = 2500  # FDA 섹션 하나에서 AI에게 넘길 최대 글자 수


def _chunks(text: str, size: int = 400) -> list[str]:
    """문장 경계에서 약 size자씩 자른다."""
    out, cur = [], ""
    for sent in re.split(r"(?<=[.;)])\s+", text):
        if cur and len(cur) + len(sent) > size:
            out.append(cur)
            cur = ""
        cur = f"{cur} {sent}".strip()
    return out + ([cur] if cur else [])


def select_passages(text: str, focus: list[str], budget: int = SECTION_BUDGET) -> str:
    """긴 문서에서 질문과 관련된 조각만 골라 budget 글자 안에 담는다.

    FDA 라벨 섹션은 2만 자가 넘기도 해서 앞부분만 자르면 뒤쪽 내용(예: 위암 적응증)을 못 본다.
    질문의 핵심어(예: gastric)가 들어간 조각을 먼저 고르고, 남는 자리는 앞쪽 조각으로 채운 뒤
    원래 순서대로 이어 붙인다. 핵심어가 없으면 기존처럼 앞에서부터 자른다.
    """
    if len(text) <= budget:
        return text
    words = {w for f in focus for w in re.findall(r"[a-z]{4,}", f.lower())}
    chunks = _chunks(text)
    scores = [sum(c.lower().count(w) for w in words) for c in chunks]
    picked, used = set(), 0
    order = sorted(range(len(chunks)), key=lambda i: (-scores[i], i)) if any(scores) else range(len(chunks))
    for i in order:
        if used + len(chunks[i]) > budget:
            continue
        picked.add(i)
        used += len(chunks[i]) + 5
    return " … ".join(chunks[i] for i in sorted(picked))


def search_fda_label(drug: str, sections: list | None = None, focus: list | None = None) -> list[Doc]:
    """sections: 필요한 라벨 섹션만 (IND 적응증 / DOSE 용법·용량 / BOX 박스 경고). None이면 전부.
    focus: 질문의 핵심어(영문). 긴 섹션에서 관련 부분을 고를 때 쓴다."""
    r = _get(OPENFDA, {"search": f'openfda.generic_name:"{drug}" OR openfda.brand_name:"{drug}"', "limit": 5})
    if r.status_code == 404:
        return []
    results = r.json().get("results", [])
    if not results:
        return []
    # 복합제보다 단일 성분 제품을 우선 (generic_name 단어 수가 적은 것)
    label = min(results, key=lambda x: len(" ".join(x.get("openfda", {}).get("generic_name", ["x" * 99])).split()))
    set_id = label.get("set_id", "")
    of = label.get("openfda", {})
    name = f"{', '.join(of.get('brand_name', []))} ({', '.join(of.get('generic_name', []))})"
    docs = []
    for code, key in LABEL_SECTIONS.items():
        if key in label and (not sections or code in sections):
            docs.append(Doc(
                id=f"FDA:{set_id[:8]}-{code}", source="fda", title=f"{name} – FDA label: {key}",
                url=f"https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid={set_id}",
                text=select_passages(" ".join(label[key]), focus or []),
                meta={"section": key, "set_id": set_id},
            ))
    return docs
