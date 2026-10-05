from bio_gpt.sources.fda import _pick_label, select_passages


def test_select_passages_prefers_focus_words():
    text = " ".join(f"Sentence {i} about {'gastric cancer' if i == 150 else 'melanoma'} dosing." for i in range(300))
    out = select_passages(text, ["gastric cancer"], budget=500)
    assert "gastric" in out and len(out) <= 520


def test_short_text_untouched():
    assert select_passages("short", ["x"]) == "short"


def test_single_ingredient_label_preferred():
    combo = {"openfda": {"generic_name": ["metformin and sitagliptin"]}}
    single = {"openfda": {"generic_name": ["metformin"]}}
    assert _pick_label([combo, single]) is single
