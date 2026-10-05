"""설계서(architecture.md) → PDF 변환.

Markdown을 HTML로 바꾸고(Mermaid 구조도 포함), Chrome 헤드리스로 A4 PDF를 인쇄한다.
실행: uv run --with markdown python docs/build_pdf.py
"""
import re
import subprocess
from pathlib import Path

import markdown

DOCS = Path(__file__).resolve().parent
SRC = DOCS / "architecture.md"
HTML = DOCS / "architecture.html"
PDF = DOCS / "Bio-GPT_프로토타입_설계서.pdf"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
body { font-family: "Apple SD Gothic Neo", "Pretendard", sans-serif; font-size: 10.5pt; line-height: 1.7; color: #1f2328; }
.cover { height: 250mm; display: flex; flex-direction: column; justify-content: center; page-break-after: always; }
.cover .kicker { color: #0b6e4f; font-weight: 700; letter-spacing: .05em; font-size: 11pt; }
.cover h1 { font-size: 26pt; line-height: 1.3; margin: 8mm 0 4mm; border: none; }
.cover .sub { font-size: 13pt; color: #555; }
.cover .meta { margin-top: 30mm; font-size: 10pt; color: #555; border-top: 2px solid #0b6e4f; padding-top: 4mm; }
h1 { font-size: 17pt; border-bottom: 2px solid #0b6e4f; padding-bottom: 2mm; }
h2 { font-size: 14pt; color: #0b6e4f; margin-top: 9mm; border-bottom: 1px solid #d0d7de; padding-bottom: 1.5mm; page-break-after: avoid; }
h2.newpage { page-break-before: always; margin-top: 0; }
h3 { font-size: 11.5pt; margin-top: 6mm; page-break-after: avoid; }
table { border-collapse: collapse; width: 100%; margin: 3mm 0; font-size: 8.6pt; page-break-inside: auto; }
tr { page-break-inside: avoid; }
th, td { border: 1px solid #d0d7de; padding: 1.6mm 2mm; vertical-align: top; text-align: left; }
th { background: #eef6f2; }
code { font-family: Menlo, monospace; font-size: 8.5pt; background: #f3f4f6; padding: 0 1mm; border-radius: 2px; }
pre { background: #f6f8fa; padding: 3mm; border-radius: 4px; font-size: 7pt; line-height: 1.45; white-space: pre-wrap; page-break-inside: avoid; }
pre code { background: none; padding: 0; }
blockquote { margin: 0 0 4mm; padding: 2mm 4mm; border-left: 3px solid #0b6e4f; background: #f3faf6; color: #444; }
blockquote table { background: #fff; }
div.cs { border-left: 3px solid #2f5fb3; background: #f2f6fd; padding: 1mm 4mm 2mm; margin: 5mm 0; border-radius: 4px; }
div.cs > p:first-child strong { color: #2f5fb3; font-size: 11pt; }
div.cs pre { background: #fff; border: 1px solid #dbe4f3; }
div.cs table { background: #fff; }
div.cs th { background: #e6edf9; }
table { word-break: keep-all; }
div.cs td:first-child, div.cs th:first-child { min-width: 24mm; width: 44%; }
div.cs td:first-child code { font-weight: 400; }
.mermaid { text-align: center; margin: 4mm 0; page-break-inside: avoid; }
.mermaid svg { max-height: 200mm; max-width: 100%; height: auto; }
hr { border: none; border-top: 1px solid #d0d7de; margin: 6mm 0; }
"""


SCRIPT_CSS = """
body { font-size: 13.5pt; line-height: 1.75; }
h2 { font-size: 17pt; margin-top: 10mm; }
h3 { font-size: 14pt; }
p { margin: 0 0 4mm; }
"""


def build(src: Path = SRC, pdf: Path = PDF, with_cover: bool = True, script: bool = False) -> None:
    md = src.read_text()
    if with_cover:
        # 첫 제목과 인용 블록은 표지로 따로 만든다
        md = re.sub(r"^# .*\n", "", md, count=1)
        md = re.sub(r"^(> .*\n)+", "", md.lstrip(), count=1)
    # 목록 앞에 빈 줄이 없으면 Markdown이 목록으로 인식하지 못하므로 빈 줄을 넣는다
    md = re.sub(r"(?m)^((?![ \t]*(?:[-*] |\||>|\d+\. )).+)\n(?=[ \t]*(?:- |\d+\. ))", r"\1\n\n", md)
    exts = ["tables", "fenced_code", "sane_lists", "md_in_html"] + (["nl2br"] if script else [])
    body = markdown.markdown(md, extensions=exts)
    body = re.sub(r'<pre><code class="language-mermaid">(.*?)</code></pre>',
                  lambda m: f'<div class="mermaid">{m.group(1)}</div>', body, flags=re.S)
    body = re.sub(r"<hr ?/?>", "", body)  # 장 구분은 페이지 나눔으로 대신
    # 큰 장은 새 페이지에서 시작
    body = re.sub(r"<h2>(\d\.|부록 A)", r'<h2 class="newpage">\1', body)
    cover = """<div class="cover">
      <div class="kicker">AI 에이전트 핵심기능 프로토타입 개발</div>
      <h1>Bio-GPT<br>핵심기능 프로토타입 설계서</h1>
      <div class="sub">출처를 달고, 스스로 검사하는 바이오·제약 연구용 AI 질의응답 서비스</div>
      <div class="meta">시스템 구조 · AI 모델 연결과 프롬프트 설계 · LangGraph 프로토타입 · 평가<br>작성일 2026. 10. 4.</div>
    </div>"""
    if not with_cover:
        cover = ""
        body = body.replace('<h2 class="newpage">', "<h2>")
    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>Bio-GPT 설계서</title>
<style>{CSS}{SCRIPT_CSS if script else ""}</style>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script>mermaid.initialize({{startOnLoad: true, theme: "neutral", flowchart: {{htmlLabels: true}}, fontFamily: "Apple SD Gothic Neo"}});</script>
</head><body>{cover}{body}</body></html>"""
    HTML.write_text(html)
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--virtual-time-budget=15000", f"--print-to-pdf={pdf}", HTML.as_uri()],
                   check=True, capture_output=True)
    HTML.unlink()
    print(f"저장: {pdf}")


if __name__ == "__main__":
    import sys
    if "notes" in sys.argv[1:]:   # 발표 메모: uv run --with markdown python docs/build_pdf.py notes
        build(DOCS / "presentation_script.md", DOCS / "Bio-GPT_발표대본.pdf", with_cover=False, script=True)
        build(DOCS / "presentation_qa.md", DOCS / "Bio-GPT_발표준비_예상질문.pdf", with_cover=False)
    else:
        build()
