"""영-한 의학 표준 용어 사전 (SFR-003 경량화: 용어 사전 매핑).

1차 평가에서 '흑색종'을 '멜라노마'로, 'mesothelioma'를 엉뚱한 말로 옮기는 문제가 나와 추가했다.
  - EN_KO: 근거 문서에 등장한 영문 용어만 골라 생성 프롬프트에 [용어 표기 기준]으로 넣는다.
  - VARIANTS: 모델이 자주 쓰는 비표준 한국어 표기를 표준어로 자동 교정한다.
대한의학회 의학용어 표기 관행을 참고해 정리한 초안이며, 운영 전 전문가 검수가 필요하다.
"""
import re

EN_KO = {
    # 종양
    "melanoma": "흑색종",
    "non-small cell lung cancer": "비소세포폐암",
    "small cell lung cancer": "소세포폐암",
    "NSCLC": "비소세포폐암",
    "mesothelioma": "중피종",
    "head and neck squamous cell carcinoma": "두경부 편평세포암",
    "squamous cell carcinoma": "편평세포암",
    "hepatocellular carcinoma": "간세포암",
    "renal cell carcinoma": "신세포암",
    "urothelial carcinoma": "요로상피암",
    "colorectal cancer": "대장암",
    "gastric cancer": "위암",
    "gastric or gastroesophageal junction adenocarcinoma": "위 또는 위식도접합부 선암",
    "endometrial cancer": "자궁내막암",
    "esophageal cancer": "식도암",
    "cervical cancer": "자궁경부암",
    "endometrial carcinoma": "자궁내막암",
    "ovarian cancer": "난소암",
    "breast cancer": "유방암",
    "triple-negative breast cancer": "삼중음성 유방암",
    "prostate cancer": "전립선암",
    "pancreatic cancer": "췌장암",
    "biliary tract cancer": "담도암",
    "hodgkin lymphoma": "호지킨림프종",
    "lymphoma": "림프종",
    "leukemia": "백혈병",
    "multiple myeloma": "다발골수종",
    "solid tumor": "고형암",
    "solid tumors": "고형암",
    "metastatic": "전이성",
    "unresectable": "절제 불가능한",
    "adjuvant": "수술 후 보조요법",
    "neoadjuvant": "수술 전 보조요법",
    "first-line": "1차 치료",
    "overall survival": "전체 생존기간",
    "progression-free survival": "무진행 생존기간",
    "objective response rate": "객관적 반응률",
    "microsatellite instability-high": "고빈도 현미부수체 불안정성(MSI-H)",
    # 폐·면역 관련 이상반응
    "interstitial lung disease": "간질성 폐질환",
    "pneumonitis": "폐렴(폐장염)",
    "cytokine release syndrome": "사이토카인 방출 증후군",
    "neurotoxicity": "신경독성",
    "embryo-fetal toxicity": "배아-태아 독성",
    "immune-mediated": "면역 매개",
    "infusion-related reactions": "주입 관련 반응",
    "hepatotoxicity": "간독성",
    "lactic acidosis": "젖산산증",
    # 신경·대사
    "alzheimer's disease": "알츠하이머병",
    "alzheimer disease": "알츠하이머병",
    "mild cognitive impairment": "경도인지장애",
    "clinical dementia rating": "임상치매평가척도(CDR)",
    "clinical dementia rating-sum of boxes": "임상치매평가척도 합산점수(CDR-SB)",
    "dementia": "치매",
    "amyloid": "아밀로이드",
    "amyloid-related imaging abnormalities": "아밀로이드 관련 영상 이상(ARIA)",
    "type 2 diabetes mellitus": "제2형 당뇨병",
    "type 2 diabetes": "제2형 당뇨병",
    "glycemic control": "혈당 조절",
    "obesity": "비만",
    "renal impairment": "신장애",
    "hypertension": "고혈압",
    # 임상시험 용어
    "randomized": "무작위배정",
    "placebo": "위약",
    "double-blind": "이중눈가림",
    "open-label": "공개",
    "recruiting": "모집 중",
    "not yet recruiting": "모집 전",
    "active, not recruiting": "진행 중(모집 종료)",
    "active_not_recruiting": "진행 중(모집 종료)",
    "not_yet_recruiting": "모집 전",
    "terminated": "조기 종료",
    "completed": "완료",
    "withdrawn": "철회",
    "enrollment": "등록 환자 수",
    "boxed warning": "박스 경고",
    "indications and usage": "적응증",
    "dosage and administration": "용법·용량",
    "pharmacokinetics": "약동학",
}

# 모델이 쓰는 비표준 표기 → 표준 표기
VARIANTS = {
    "멜라노마": "흑색종",
    "메소텔리오마": "중피종",
    "중피세포종": "중피종",
    "폐 간질 질환": "간질성 폐질환",
    "간질 폐질환": "간질성 폐질환",
    "간질성 폐 질환": "간질성 폐질환",
    "사이토카인 릴리즈 증후군": "사이토카인 방출 증후군",
    "알츠하이머 질환": "알츠하이머병",
    "플라시보": "위약",
    "플라세보": "위약",
    "랜덤화": "무작위배정",
    "고급 비소세포폐암": "진행성 비소세포폐암",
    "고급 고형암": "진행성 고형암",
    "고급 실체 종": "진행성 고형암",
    "네오아드주반트": "수술 전 보조요법",
    "네오아쥬반트": "수술 전 보조요법",
    "어주반트": "수술 후 보조요법",
    "아쥬반트": "수술 후 보조요법",
}

# 한국어 답변에 섞이면 안 되는 문자 (키릴, 한자, 일본 가나, 태국 문자 등)
# 키릴(러시아), 아랍, 데바나가리, 태국, 일본 가나, 한자
FOREIGN_SCRIPT_RE = re.compile(r"[\u0400-\u04FF\u0600-\u06FF\u0900-\u097F\u0E00-\u0E7F\u3040-\u30FF\u4E00-\u9FFF]+")


def terms_in(text: str) -> dict:
    """문서에 실제 등장한 영문 용어만 골라 반환 (프롬프트 길이 절약)."""
    low = text.lower()
    found = {en: ko for en, ko in EN_KO.items() if re.search(rf"(?<![a-z]){re.escape(en.lower())}(?![a-z])", low)}
    # 긴 용어에 포함되는 짧은 용어는 뺀다 (예: 'lymphoma' ⊂ 'hodgkin lymphoma')
    return {en: ko for en, ko in found.items() if not any(en != o and en.lower() in o.lower() for o in found)}


def normalize(text: str) -> str:
    for bad, good in VARIANTS.items():
        text = text.replace(bad, good)
    return text
