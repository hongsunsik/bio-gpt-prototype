# Bio-GPT 프로토타입

생성형 AI 기반 바이오·제약 R&D 지식 플랫폼(Bio-GPT)의 **1차년도 핵심기능 프로토타입**입니다.
연구자의 자연어 질문을 분석해 PubMed · ClinicalTrials.gov · openFDA를 검색하고, **모든 문장에 출처를 단 답변**을 만든 뒤, 답변이 근거와 일치하는지 **자동으로 검증**합니다.

- 설계서(구조, 코드별 선택 과정과 근거, LLM 선택·Gemini 비교, 프롬프트 설계, 평가, 용어 설명): [`docs/architecture.md`](docs/architecture.md) · PDF: [`docs/Bio-GPT_프로토타입_설계서.pdf`](docs/Bio-GPT_프로토타입_설계서.pdf)
- RFP 피드백 반영 기록(범위 축소, 추적표, 측정 조건, 예산 역산): [`docs/rfp_feedback_notes.md`](docs/rfp_feedback_notes.md)
- 골든셋 평가 리포트: [`eval/results/`](eval/results/)

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

```bash
# 1) 로컬 LLM 준비 (Ollama)
ollama pull qwen3:8b
ollama create bio-qwen3 -f Modelfile.qwen3-bio   # 문맥 16K 설정

# 2) 웹 UI
uv sync
uv run streamlit run app.py

# 3) 골든셋 평가
uv run python eval/run_eval.py
```

클라우드 모델을 쓰려면 `.env.example`을 `.env`로 복사하고 값을 채우면 됩니다(코드 수정 불필요).

## 구조

```
app.py                 Streamlit UI
bio_gpt/
  graph.py             LangGraph 워크플로 (guard → plan → retrieve → generate ⇄ verify → finalize)
  sources.py           PubMed / ClinicalTrials.gov / openFDA 연동
  prompts.py           프롬프트 명세 (버전 관리)
  glossary.py          영-한 의학 표준 용어 사전
  llm.py               OpenAI 호환 LLM 클라이언트 (Ollama · GPT-4o · Gemini 교체 가능)
  store.py             SQLite 로그·피드백
eval/
  golden_set.jsonl     골든셋 16문항
  run_eval.py          지표 계산 + 기준선 판정 리포트
docs/architecture.md   시스템 설계서
```

> 의료적 판단을 대신하지 않는 교육용 프로토타입입니다. 사업명·예산 등 RFP 수치는 가상 값입니다.
