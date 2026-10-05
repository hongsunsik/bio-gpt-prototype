from langchain_core.documents import Document

from bio_gpt.rag.retrieval import KeywordIndex, rrf_fuse


def doc(cid, text=""):
    return Document(page_content=text, metadata={"cite_id": cid})


def test_rrf_rewards_docs_ranked_by_both():
    a, b, c = doc("a"), doc("b"), doc("c")
    fused = rrf_fuse([[a, b, c], [b, c]], k=2)
    assert [d.metadata["cite_id"] for d in fused] == ["b", "c"]


def test_keyword_index_ignores_zero_scores():
    ix = KeywordIndex([doc("1", "colitis corticosteroids"), doc("2", "melanoma dosing"), doc("3", "hepatitis")])
    assert [d.metadata["cite_id"] for d in ix.search("colitis", 3)] == ["1"]
