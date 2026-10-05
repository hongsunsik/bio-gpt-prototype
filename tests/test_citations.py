from bio_gpt.agent import citations as C


def test_repair_malformed_citations(docs):
    out = C.repair_citations("- 용량 [FDA:9333c79b-1.1]\n- 반응률 [PMID 123456]\n- 본문 (NCT01234567)", docs)
    assert out == "- 용량 [FDA:9333c79b-IND]\n- 반응률 [PMID:123456]\n- 본문 [NCT01234567]"


def test_unknown_citation_left_as_is(docs):
    assert C.repair_citations("- 없음 [PMID:999]", docs) == "- 없음 [PMID:999]"


def test_not_found_line_loses_citation(docs):
    assert C.repair_citations("- 확인되지 않음 [PMID:123456]", docs) == "- 확인되지 않음"


def test_join_wrapped():
    assert C.join_wrapped("- 대장염 1.7%\n  그중 1.1% [DOC:p12-3]\n- 다음") == "- 대장염 1.7% 그중 1.1% [DOC:p12-3]\n- 다음"


def test_claim_lines():
    assert C.claim_lines("요약\n- a\n본문 [NCT01234567]") == ["- a", "본문 [NCT01234567]"]


def test_runaway():
    assert C.is_runaway("같은구절반복해요" * 8)
    assert not C.is_runaway("- 정상적인 답변입니다 [PMID:1]")
