from bio_gpt.agent import writer


def test_echoes_reason():
    assert writer.echoes_reason("주장에서 틀림", "")
    assert writer.echoes_reason("문서에는 1.7%로 나와 있어 다름", "문서에는 1.7%로 나와 있어 다름 확인")
    assert not writer.echoes_reason("정상 문장 [PMID:1]", "전혀 다른 지적 내용입니다 길게")


def test_repair_keeps_passed_and_drops_bad_fixes(docs, monkeypatch):
    ok = {"has_citation": True, "citation_exists": True}
    rows = [{"n": 1, "line": "- 통과 [PMID:123456]", "checks": ok},
            {"n": 2, "line": "- 틀림 [PMID:999]", "checks": {"has_citation": True, "citation_exists": False}},
            {"n": 3, "line": "- 출처 없음", "checks": {"has_citation": False}},
            {"n": 4, "line": "- 베낌 [NCT01234567]", "checks": {**ok, "supported": False}, "reason": "주장 불일치"}]
    fixes = {"fixed": [{"n": 2, "line": "고친 줄 [NCT01234567]"}, {"n": 3, "line": "여전히 출처 없음"},
                       {"n": 4, "line": "주장에서 다름 [NCT01234567]"}]}
    monkeypatch.setattr(writer, "chat_json", lambda *a, **k: fixes)
    assert writer.repair(rows, docs) == "- 통과 [PMID:123456]\n- 고친 줄 [NCT01234567]"


def test_cot_split():
    assert writer._split_cot("### 생각 과정\n생각\n### 답변\n- 답 [PMID:1]") == ("생각", "- 답 [PMID:1]")


def test_runaway_answer_discarded(docs, monkeypatch):
    monkeypatch.setattr(writer, "complete", lambda *a, **k: "반복반복반복반복" * 10)
    out = writer.answer_from_docs("q", docs, "baseline")
    assert out["runaway"] and out["answer"] == ""
