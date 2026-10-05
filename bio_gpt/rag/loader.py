"""PDF 읽기와 청크 분할. 청크마다 인용 ID(DOC:p<쪽>-<번호>)를 붙인다."""
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_split(pdf_path: str, chunk_size: int, chunk_overlap: int) -> tuple[list[Document], int]:
    pages = PyMuPDFLoader(pdf_path).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    per_page: dict[int, int] = {}
    for c in chunks:
        p = c.metadata["page"]
        per_page[p] = per_page.get(p, 0) + 1
        c.metadata["cite_id"] = f"DOC:p{p + 1}-{per_page[p]}"  # DOC:p12-3 = 12쪽의 3번째 청크
    return chunks, len(pages)
