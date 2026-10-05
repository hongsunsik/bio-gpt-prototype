"""규칙 기반 안전 점검 (SER-001 프롬프트 인젝션, SFR-006 컴플라이언스).

LLM을 부르기 전에 정규식으로 먼저 거른다. 빠르고, 결과가 항상 같고, 모델에 따라 흔들리지 않는다.
"""
import re
import time

from .state import State, add_trace

INJECTION_PATTERNS = [
    r"(이전|위의?|모든)\s*(지시|명령|규칙).{0,6}(무시|잊)",
    r"ignore (all |the )?(previous|above) (instructions|prompts?)",
    r"시스템\s*프롬프트", r"system prompt", r"너의\s*(설정|지시문)", r"developer mode", r"jailbreak",
]

# 개인 복용 상담: 1인칭/가족 + 복용 여부·용량
PERSONAL_PATTERNS = [
    r"(제가|저는|저도|내가|나는|우리\s*(엄마|아빠|아이|어머니|아버지)|저희\s*(엄마|아빠|아이|어머니|아버지))",
    r"(먹어도|복용해도|맞아도|끊어도)\s*(되|괜찮)",
    r"몇\s*(mg|밀리|알|정).{0,6}(먹|복용)",
]

# 질문 분석 모델이 personal_medical로 분류했을 때, 이 표현이 없으면 일반 정보 질문으로 되돌린다
PERSONAL_HINT = (r"(제가|저는|저도|저희|내가|나는|우리\s*(엄마|아빠|아이|애|어머니|아버지|가족)"
                 r"|엄마|아빠|어머니|아버지|남편|아내|와이프|할머니|할아버지|아들|딸|친구)")

# 의약품 광고·허가 외 사용 유도로 보일 수 있는 표현
COMPLIANCE_PATTERNS = [r"완치", r"부작용이?\s*(전혀\s*)?없", r"100\s*%\s*(효과|안전)", r"최고의\s*(약|치료)",
                       r"허가\s*외.{0,6}권장"]


def is_injection(text: str) -> bool:
    return any(re.search(p, text, re.I) for p in INJECTION_PATTERNS)


def is_personal_medical(text: str) -> bool:
    return any(re.search(p, text) for p in PERSONAL_PATTERNS)


def mentions_person(text: str) -> bool:
    return bool(re.search(PERSONAL_HINT, text))


def compliance_flags(text: str) -> list[str]:
    return [p for p in COMPLIANCE_PATTERNS if re.search(p, text)]


def guard(state: State) -> State:
    t0, q = time.time(), state["question"]
    if is_injection(q):
        return {"category": "blocked", "blocked_reason": "injection",
                "trace": add_trace(state, "guard", t0, "프롬프트 인젝션 패턴 차단")}
    if is_personal_medical(q):
        return {"category": "personal_medical", "trace": add_trace(state, "guard", t0, "개인 의료 상담 패턴 감지")}
    return {"category": "", "trace": add_trace(state, "guard", t0, "통과")}
