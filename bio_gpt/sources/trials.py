"""ClinicalTrials.gov API v2 임상시험 검색."""
from ..models import Doc
from .http import get

CTGOV = "https://clinicaltrials.gov/api/v2/studies"
MAX_CHARS = 1000


def _to_doc(study: dict) -> Doc:
    p = study.get("protocolSection", {})
    ident = p.get("identificationModule", {})
    status = p.get("statusModule", {})
    design = p.get("designModule", {})
    nct = ident.get("nctId", "")
    title = ident.get("briefTitle", "")
    conditions = ", ".join(p.get("conditionsModule", {}).get("conditions", []))
    interventions = ", ".join(i.get("name", "") for i in p.get("armsInterventionsModule", {}).get("interventions", []))
    phases = ", ".join(design.get("phases", []) or ["N/A"])
    enrollment = design.get("enrollmentInfo", {}).get("count", "")
    summary = p.get("descriptionModule", {}).get("briefSummary", "")
    text = (f"{title}. Status: {status.get('overallStatus', '')}. Phase: {phases}. "
            f"Conditions: {conditions}. Interventions: {interventions}. Enrollment: {enrollment}. "
            f"Start: {status.get('startDateStruct', {}).get('date', '')}. Summary: {summary}")
    return Doc(
        id=nct, source="trials", title=title,
        url=f"https://clinicaltrials.gov/study/{nct}", text=text[:MAX_CHARS],
        meta={"status": status.get("overallStatus", ""), "phase": phases},
    )


def search_trials(query: str, n: int = 4) -> list[Doc]:
    data = get(CTGOV, {"query.term": query, "pageSize": n, "format": "json"}).json()
    return [_to_doc(s) for s in data.get("studies", [])]
