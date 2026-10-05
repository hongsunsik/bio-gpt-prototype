# Bio-GPT 프로토타입

생성형 AI 기반 바이오·제약 R&D 지식 플랫폼(Bio-GPT)의 **1차년도 핵심기능 프로토타입**입니다.
연구자의 자연어 질문을 분석해 PubMed · ClinicalTrials.gov · openFDA를 검색하고, **모든 문장에 출처를 단 답변**을 만든 뒤, 답변이 근거와 일치하는지 **자동으로 검증**합니다.

- 설계서(구조, 코드별 선택 과정과 근거, LLM 선택·Gemini 비교, 프롬프트 설계, 평가, 용어 설명): [`docs/architecture.md`](docs/architecture.md) · PDF: [`docs/Bio-GPT_프로토타입_설계서.pdf`](docs/Bio-GPT_프로토타입_설계서.pdf)
- RFP 피드백 반영 기록(범위 축소, 추적표, 측정 조건, 예산 역산): [`docs/rfp_feedback_notes.md`](docs/rfp_feedback_notes.md)
- 골든셋 평가 리포트: [`eval/results/`](eval/results/)
- **2주차: 업로드 문서 RAG + 파라미터·프롬프트 실험**: [`docs/rag_experiments.md`](docs/rag_experiments.md)

## 2주차: 업로드 문서(PDF) RAG

PDF를 올리면 가이드의 8단계(로드 → 분할 → 임베딩 → FAISS → 검색기 → 프롬프트 → LLM → 출력)로 답하고, 기존 Bio-GPT의 **출처 표시·자동 검증**을 그대로 적용합니다.
사이드바에서 `chunk_size`, `chunk_overlap`, `k`, 검색 방식(similarity / mmr / hybrid), 프롬프트 기법을 바꿔 결과를 바로 비교할 수 있습니다.

| | 시작점: 예제 코드 그대로 | 최종 |
|---|---|---|
| 설정 | 400 / 100 / k=3 / similarity, 예제 프롬프트 | 800 / 0 / k=5 / **hybrid**(의미+키워드 검색), **Few-shot** 프롬프트 + 답 검증 |
| 정답 근거를 찾아온 비율 | 58% | 83% |
| 최종 정답률 (키트루다 FDA 라벨 106쪽, 13문항) | 69% | **77%** |
| 출처 없는 문장 | 17줄 | **0줄** |

프롬프트 기법 6가지(예제 · 역할+규칙 · 형식 지정 · Few-shot · CoT · Few-shot+CoT)와 검색 설정 192가지 조합의 비교, 실험 중 고친 버그 4개는 [`docs/rag_experiments.md`](docs/rag_experiments.md)에 있습니다.

## 핵심 기능

| 기능 | 요구사항 |
|---|---|
| 질의 분류(연구 / 개인 의료 / 무관), 검색 모듈 자동 선택, 영문 검색어 재작성, 개체 추출 | SFR-002, SFR-003 |
| 논문·임상·FDA 라벨 병렬 검색, 재시도·부분 실패 허용 | SFR-004, SIR-001 |
| 줄마다 출처 ID(PMID / NCT / FDA) 인용, 근거 없으면 '확인되지 않음' | SFR-005 |
| 인용 실재 · 숫자 일치 · LLM 사실 판정 3단계 검증 → 재생성 → 불일치 문장 제거·답변 보류 | SFR-006 |
| 개인 의료 상담·프롬프트 인젝션 차단, 면책 문구 상시 표시 | SFR-006, SER-001 |
| Like/Dislike·코멘트, 질의·근거·검증 로그, 골든셋 자동 평가 | SFR-007, SIR-002, TER-001 |

## 실행

### A. OpenAI 키로 실행 (2주차 실습 가이드 방식, Windows/Mac 공통)

```bash
python -m venv venv
venv\Scripts\activate          # Mac: source venv/bin/activate
pip install -r requirements.txt
# 프로젝트 폴더에 .env 파일을 만들고 한 줄:  OPENAI_API_KEY=sk-...
streamlit run app.py
```

`.env`에 `OPENAI_API_KEY`만 있으면 GPT-4o + `text-embedding-3-small`로 동작합니다. 사이드바에서 "샘플: 키트루다 FDA 허가 라벨"을 체크하거나 PDF를 올리세요.

### B. 로컬 모델로 실행 (외부 전송 없음)

macOS에서는 `scripts/start_mac.command`를 더블클릭하면 Ollama 켜기 → 모델 확인 → 예열 → 웹 화면 열기까지 한 번에 진행합니다.
직접 하려면:

```bash
ollama pull qwen3:8b && ollama pull bge-m3
ollama create bio-qwen3 -f Modelfile.qwen3-bio   # 문맥 16K 설정
uv sync
uv run streamlit run app.py
```

### 평가

```bash
uv run python eval/run_eval.py                    # 1차: 외부 DB 검색 골든셋 16문항
uv run python eval/rag_sweep.py retrieval         # 2주차: 검색 파라미터 실험 (그 밖의 단계는 docs/rag_experiments.md)
```

## 구조

```
app.py                 Streamlit UI
bio_gpt/
  graph.py             LangGraph 워크플로 (guard → plan → retrieve → generate ⇄ verify → finalize)
  sources.py           PubMed / ClinicalTrials.gov / openFDA 연동
  prompts.py           프롬프트 명세 (버전 관리)
  glossary.py          영-한 의학 표준 용어 사전
  llm.py               OpenAI 호환 LLM 클라이언트 (Ollama · GPT-4o · Gemini 교체 가능)
  rag_module.py        업로드 PDF RAG: 분할 · 임베딩 · FAISS · 검색(similarity/mmr/hybrid)
  store.py             SQLite 로그·피드백
eval/
  golden_set.jsonl     골든셋 16문항
  run_eval.py          지표 계산 + 기준선 판정 리포트
  pdf_golden_set.jsonl 업로드 문서 RAG 평가 13문항 (정답 원문·키워드)
  rag_sweep.py         chunk_size · overlap · k · 검색 방식 · 프롬프트 기법 비교 실험
samples/               실험·시연용 PDF (키트루다 FDA 허가 라벨)
docs/architecture.md   시스템 설계서
docs/rag_experiments.md 2주차 RAG 실험 보고서
```

> 의료적 판단을 대신하지 않는 교육용 프로토타입입니다. 사업명·예산 등 RFP 수치는 가상 값입니다.
