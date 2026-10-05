import pytest

from bio_gpt.models import Doc


@pytest.fixture
def docs() -> list[Doc]:
    return [
        Doc("PMID:123456", "pubmed", "t", "u", "Melanoma response rate 45.2% in 1,234 patients."),
        Doc("NCT01234567", "trials", "t", "u", "Phase 3 trial enrolled 500 patients"),
        Doc("FDA:9333c79b-IND", "fda", "t", "u", "Indicated for melanoma. 200 mg every 3 weeks"),
        Doc("DOC:p12-3", "pdf", "t", "", "colitis occurred in 1.7% of patients; 1.1% required corticosteroids"),
    ]
