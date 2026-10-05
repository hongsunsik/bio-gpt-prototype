from bio_gpt.agent import planner


class FakeIndex:
    text = "keytruda pembrolizumab melanoma colitis"

    class store:
        @staticmethod
        def similarity_search_with_score(q, k=1):
            return [(None, 0.5 if "면역" in q else 0.9)]


def run_plan(monkeypatch, llm_out: dict, question: str, ix=None):
    monkeypatch.setattr(planner, "chat_json", lambda *a, **k: dict(llm_out))
    return planner.plan({"question": question, "trace": [], "doc_index": ix})


def test_off_topic_with_drug_becomes_research(monkeypatch):
    out = run_plan(monkeypatch, {"category": "off_topic", "entities": [{"normalized": "aspirin", "type": "drug"}],
                                 "modules": ["pubmed"]}, "아스피린이 뭐야")
    assert out["category"] == "research"


def test_personal_without_person_becomes_research(monkeypatch):
    out = run_plan(monkeypatch, {"category": "personal_medical", "modules": ["fda"]}, "소아 환자의 허가 용량은?")
    assert out["category"] == "research"
    out = run_plan(monkeypatch, {"category": "personal_medical"}, "저는 이 약 먹어도 되나요")
    assert out["category"] == "personal_medical"


def test_empty_plan_falls_back_to_pubmed(monkeypatch):
    out = run_plan(monkeypatch, {"category": "research", "rewritten_question": "GLP-1"}, "GLP-1 연구")
    assert out["modules"] == ["pubmed"] and out["queries"] == {"pubmed": "GLP-1"}


def test_unknown_modules_and_sections_dropped(monkeypatch):
    out = run_plan(monkeypatch, {"category": "research", "modules": ["fda", "web"], "fda_sections": ["IND", "XX"]}, "q")
    assert out["modules"] == ["fda"] and out["fda_sections"] == ["IND"]


def test_routes_to_doc_only_when_drug_in_doc(monkeypatch):
    ents = [{"normalized": "pembrolizumab", "type": "drug"}]
    assert run_plan(monkeypatch, {"category": "research", "entities": ents}, "펨브롤리주맙 대장염", FakeIndex())["modules"] == ["pdf"]
    ents = [{"normalized": "aspirin", "type": "drug"}]
    assert run_plan(monkeypatch, {"category": "research", "entities": ents, "modules": ["pubmed"]},
                    "아스피린 효과", FakeIndex())["modules"] == ["pubmed"]


def test_about_doc_rules():
    ix = FakeIndex()
    assert planner.is_about_doc("이 약의 금기는?", [], ix)  # 지시어
    assert planner.is_about_doc("흑색종 치료", [{"normalized": "melanoma", "type": "disease"}], ix)
    assert planner.is_about_doc("면역 부작용 비율", [], ix)  # 거리 0.5
    assert not planner.is_about_doc("부작용 비율", [], ix)  # 거리 0.9
