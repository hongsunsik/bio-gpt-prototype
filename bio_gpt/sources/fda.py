"""openFDA 의약품 허가 라벨 검색. 라벨에서 질문에 필요한 섹션(적응증·작용기전·경고 등)만 문서로 만든다."""
import re

from ..models import Doc
from .http import get

OPENFDA = "https://api.fda.gov/drug/label.json"
# 섹션 코드 → 라벨 필드 (앞의 필드가 없으면 뒤의 필드). 일반의약품(OTC) 라벨은 purpose·warnings만 있고
# 처방약 라벨은 mechanism_of_action·warnings_and_cautions가 있다.
LABEL_SECTIONS = {
    "PURPOSE": ("purpose",),
    "IND": ("indications_and_usage",),
    "MOA": ("mechanism_of_action", "clinical_pharmacology"),
    "DOSE": ("dosage_and_administration",),
    "BOX": ("boxed_warning",),
    "WARN": ("warnings_and_cautions", "warnings"),
}
DEFAULT_SECTIONS = ("IND", "DOSE", "BOX")
CANDIDATES = 20  # 라벨 후보 수. 같은 성분 제품이 수백 개라 몇 개 더 받아서 고른다
SECTION_BUDGET = 2500  # 섹션 하나에서 LLM에 넘길 최대 글자 수

# FDA는 미국식 일반명(USAN)으로 등록돼 있어서 국제명(INN)으로 찾으면 0건이 나오는 약들
US_NAMES = {"paracetamol": "acetaminophen", "adrenaline": "epinephrine", "noradrenaline": "norepinephrine",
            "salbutamol": "albuterol", "glibenclamide": "glyburide", "pethidine": "meperidine",
            "lignocaine": "lidocaine", "frusemide": "furosemide", "ciclosporin": "cyclosporine",
            "rifampicin": "rifampin", "aciclovir": "acyclovir"}


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
    """긴 섹션에서 질문 핵심어가 많이 나온 조각부터 budget 안에 담고, 원래 순서로 이어 붙인다.

    라벨 섹션은 2만 자가 넘기도 해서 앞에서부터 자르면 뒤쪽 적응증(예: 위암)을 놓친다.
    핵심어가 하나도 안 나오면 앞에서부터 채운다.
    """
    if len(text) <= budget:
        return text
    words = {w for f in focus for w in re.findall(r"[a-z]{4,}", f.lower())}
    chunks = _chunks(text)
    scores = [sum(c.lower().count(w) for w in words) for c in chunks]
    order = sorted(range(len(chunks)), key=lambda i: (-scores[i], i)) if any(scores) else range(len(chunks))
    picked, used = set(), 0
    for i in order:  # 배낭 문제의 탐욕적 근사: 점수 높은 조각부터 들어가는 만큼
        if used + len(chunks[i]) > budget:
            continue
        picked.add(i)
        used += len(chunks[i]) + 5  # 구분자 " … " 길이
    return " … ".join(chunks[i] for i in sorted(picked))


def _field(label: dict, code: str) -> str | None:
    return next((f for f in LABEL_SECTIONS[code] if f in label), None)


def _ingredients(label: dict) -> int:
    name = " ".join(label.get("openfda", {}).get("generic_name", ["x"])).lower()
    return len(re.split(r",| and |/", name))


def _pick_label(results: list[dict], codes: tuple = DEFAULT_SECTIONS) -> dict:
    """단일 성분 제품을 우선하고, 그중 필요한 섹션이 가장 많이 들어 있는 라벨을 고른다.
    마지막 기준은 generic_name 단어 수 ("ASPIRIN"이 "ASPIRIN 81 MG"보다 먼저)."""
    def rank(label: dict) -> tuple:
        coverage = sum(_field(label, c) is not None for c in codes)
        words = len(" ".join(label.get("openfda", {}).get("generic_name", ["x" * 99])).split())
        return _ingredients(label), -coverage, words
    return min(results, key=rank)


def search_fda_label(drug: str, sections: list | None = None, focus: list | None = None) -> list[Doc]:
    """sections: 필요한 섹션 코드(LABEL_SECTIONS의 키). None이면 IND·DOSE·BOX.
    focus: 질문의 영문 핵심어. 긴 섹션에서 관련 부분을 고를 때 쓴다."""
    drug = US_NAMES.get(drug.strip().lower(), drug)
    r = get(OPENFDA, {"search": f'openfda.generic_name:"{drug}" OR openfda.brand_name:"{drug}"', "limit": CANDIDATES})
    if r.status_code == 404:
        return []
    results = r.json().get("results", [])
    if not results:
        return []
    codes = tuple(sections) if sections else DEFAULT_SECTIONS
    label = _pick_label(results, codes)
    set_id = label.get("set_id", "")
    of = label.get("openfda", {})
    name = f"{', '.join(of.get('brand_name', []))} ({', '.join(of.get('generic_name', []))})"
    docs = []
    for code in LABEL_SECTIONS:
        key = _field(label, code)
        if code in codes and key:
            docs.append(Doc(id=f"FDA:{set_id[:8]}-{code}", source="fda", title=f"{name} – FDA label: {key}",
                            url=f"https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid={set_id}",
                            text=select_passages(" ".join(label[key]), focus or []),
                            meta={"section": key, "set_id": set_id}))
    return docs
