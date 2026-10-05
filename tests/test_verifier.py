import pytest

from bio_gpt.agent import respond, verifier


@pytest.fixture
def judge(monkeypatch):
    """검사관 응답을 고정한다. judge.unsupported에 {번호: 이유}를 넣는다."""
    class Judge:
        unsupported: dict = {}
        calls = 0

        def __call__(self, *a, **k):
            self.calls += 1
            return {"unsupported": [{"n": n, "reason": r} for n, r in self.unsupported.items()]}
    j = Judge()
    monkeypatch.setattr(verifier, "chat_json", j)
    return j


def run(answer, docs, passed=()):
    return verifier.verify({"answer": answer, "docs": docs, "passed_lines": list(passed), "verify_runs": [], "trace": []})


def test_all_checks_pass(docs, judge):
    v = run("- 반응률 45.2% [PMID:123456]\n- 3상에 500명 [NCT01234567]", docs)["verification"]
    assert v["passed"] and all(verifier.line_ok(r) for r in v["lines"])


@pytest.mark.parametrize("line, failed", [
    ("- 출처 없음", "has_citation"),
    ("- 없는 인용 [PMID:999]", "citation_exists"),
    ("- 9999명 [PMID:123456]", "numbers_match"),
    ("- 페메트레ксed [PMID:123456]", "clean_script"),
])
def test_rule_checks(docs, judge, line, failed):
    row = run(line, docs)["verification"]["lines"][0]
    assert row["checks"][failed] is False and not verifier.line_ok(row)


def test_judge_marks_unsupported(docs, judge):
    judge.unsupported = {2: "근거 불일치"}
    rows = run("- 45.2% [PMID:123456]\n- 500명 [NCT01234567]", docs)["verification"]["lines"]
    assert verifier.line_ok(rows[0]) and not verifier.line_ok(rows[1]) and rows[1]["reason"] == "근거 불일치"


def test_previously_passed_line_not_rejudged(docs, judge):
    run("- 45.2% [PMID:123456]", docs, passed=["- 45.2% [PMID:123456]"])
    assert judge.calls == 0


def test_judge_failure_skips_judgement(docs, monkeypatch):
    def broken(*a, **k):
        raise ValueError
    monkeypatch.setattr(verifier, "chat_json", broken)
    row = run("- 45.2% [PMID:123456]", docs)["verification"]["lines"][0]
    assert "supported" not in row["checks"]


@pytest.mark.parametrize("answer, status", [
    ("- 45.2% [PMID:123456]", "answered"),
    ("- 45.2% [PMID:123456]\n- 출처 없음", "partial"),
    ("- 출처 없음", "withheld"),
    ("- 완치 45.2% [PMID:123456]", "withheld"),
    ("- 확인되지 않음", "no_evidence"),
])
def test_final_status(docs, judge, answer, status):
    s = {"answer": answer, "docs": docs, "modules": ["pubmed"], "trace": []}
    s.update(run(answer, docs))
    assert respond.finalize(s)["status"] == status
