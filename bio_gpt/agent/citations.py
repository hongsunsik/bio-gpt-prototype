"""답변 텍스트 다루기: 인용 형식, 줄 단위 정리, 반복 폭주 감지.

검사(verifier)는 '한 줄 = 한 주장, 줄 끝에 [출처 ID]'를 전제로 한다.
여기 함수들은 모델 출력을 그 형식에 맞게 다듬는다.
"""
import re

from ..models import Doc

CITE_RE = re.compile(r"\[((?:PMID:\d+)|(?:NCT\d{8})|(?:FDA:[0-9a-f]{8}-[A-Z]+)|(?:DOC:p\d+-\d+))\]")
ID_RE = re.compile(r"PMID:?\s*\d+|NCT\d{8}|FDA:[0-9a-f]{8}-[A-Z]+|DOC:p\d+-\d+")  # 본문 속 출처 ID
BRACKET_RE = re.compile(r"\[([A-Za-z]{2,5}:? ?[^\[\]\s]{2,40})\]")  # 인용처럼 생긴 괄호 (형식이 틀린 것 포함)
BULLETS = ("-", "•", "*")
NOT_FOUND = "확인되지 않음"


def format_context(docs: list[Doc]) -> str:
    return "\n\n".join(f"<doc id=\"{d.id}\" source=\"{d.source}\">\n{d.text}\n</doc>" for d in docs)


def id_list(docs: list[Doc]) -> str:
    return ", ".join(d.id for d in docs)


def _fix_bracket(token: str, ids: list[str]) -> str | None:
    """형식이 틀린 인용 하나를 진짜 ID로. 후보가 하나로 좁혀질 때만 고친다."""
    if token in ids:
        return token
    token = re.sub(r"^(PMID|FDA)\s*:?\s*", r"\1:", token)  # "PMID 123" → "PMID:123"
    if token in ids:
        return token
    stem = re.split(r"[-_.]", token)[0]  # "FDA:9333c79b-1.1" → "FDA:9333c79b"
    candidates = [i for i in ids if i.startswith(stem) or stem.endswith(i.split(":")[-1])]
    return candidates[0] if len(candidates) == 1 else None


def repair_citations(answer: str, docs: list[Doc]) -> str:
    ids = [d.id for d in docs]
    for i in ids:
        answer = answer.replace(f"({i})", f"[{i}]")  # 소괄호로 단 출처

    def fix(m: re.Match) -> str:
        fixed = _fix_bracket(m.group(1), ids)
        return f"[{fixed}]" if fixed else m.group(0)

    answer = BRACKET_RE.sub(fix, answer)
    # '확인되지 않음'은 근거가 없다는 뜻이라 출처가 붙으면 안 된다.
    # 붙은 채로 두면 검사관이 '근거가 이 문장을 뒷받침하지 않음'으로 걸러서 맞는 답이 보류된다.
    return "\n".join(CITE_RE.sub("", l).rstrip() if NOT_FOUND in l else l for l in answer.splitlines())


def join_wrapped(answer: str) -> str:
    """모델이 한 문장을 여러 줄로 쓰면 출처가 다음 줄로 밀린다. 글머리표 없는 줄은 앞 줄에 붙인다."""
    out: list[str] = []
    for line in answer.splitlines():
        s = line.strip()
        if out and s and not s.startswith((*BULLETS, "_")) and out[-1].lstrip().startswith(BULLETS):
            out[-1] = f"{out[-1].rstrip()} {s}"
        else:
            out.append(line)
    return "\n".join(out)


def claim_lines(answer: str) -> list[str]:
    """검사 대상 줄: 글머리표로 시작하거나 인용이 있는 줄."""
    return [l.strip() for l in answer.splitlines() if l.strip().startswith(BULLETS) or CITE_RE.search(l)]


def as_bullets(answer: str) -> str:
    """서술형 답을 줄마다 글머리표를 단 형태로 (기본 프롬프트 답을 같은 기준으로 검사하기 위해)."""
    return "\n".join(f"- {l.strip()}" for l in answer.splitlines() if l.strip())


def is_runaway(text: str) -> bool:
    """같은 구절(8~40자)이 연달아 6번 이상 나오면 반복 폭주로 본다."""
    return bool(re.search(r"(.{8,40}?)\1{5,}", text, flags=re.S))
