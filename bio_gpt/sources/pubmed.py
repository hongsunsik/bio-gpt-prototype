"""PubMed 논문 검색 (NCBI E-utilities). esearch로 PMID를 받고 efetch로 초록을 가져온다."""
import xml.etree.ElementTree as ET

from ..models import Doc
from .http import get

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
MAX_CHARS = 1500


def _to_doc(article: ET.Element) -> Doc | None:
    abstract = " ".join("".join(a.itertext()) for a in article.findall(".//Abstract/AbstractText"))
    if not abstract:  # 초록이 없으면 근거로 쓸 내용이 없다
        return None
    pmid = article.findtext(".//PMID")
    title_el = article.find(".//ArticleTitle")
    title = "".join(title_el.itertext()) if title_el is not None else ""
    year = article.findtext(".//PubDate/Year") or article.findtext(".//PubDate/MedlineDate") or ""
    journal = article.findtext(".//Journal/Title") or ""
    return Doc(
        id=f"PMID:{pmid}", source="pubmed", title=title,
        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        text=f"{title} ({journal}, {year}). {abstract}"[:MAX_CHARS],
        meta={"year": year, "journal": journal},
    )


def search_pubmed(query: str, n: int = 4) -> list[Doc]:
    params = {"db": "pubmed", "term": query, "retmax": n, "sort": "relevance", "retmode": "json"}
    ids = get(f"{EUTILS}/esearch.fcgi", params).json()["esearchresult"]["idlist"]
    if not ids:
        return []
    xml = get(f"{EUTILS}/efetch.fcgi", {"db": "pubmed", "id": ",".join(ids), "retmode": "xml"}).text
    docs = (_to_doc(a) for a in ET.fromstring(xml).findall(".//PubmedArticle"))
    return [d for d in docs if d]
