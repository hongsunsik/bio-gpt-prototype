#!/bin/bash
# Bio-GPT 한 번에 켜기 (macOS). 더블클릭하면:
#   Ollama 켜기 → 모델 확인(없으면 받기) → 라이브러리 설치 → 모델 예열 → 웹 화면 열기
# 끌 때는 이 터미널 창에서 Ctrl+C (또는 창 닫기)
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
cd "$(dirname "$0")/.." || exit 1
URL="http://localhost:8501"
step() { printf "\n\033[1m▶ %s\033[0m\n" "$1"; }
fail() { printf "\n\033[31m✖ %s\033[0m\n" "$1"; read -r -p "엔터를 누르면 창이 닫힙니다"; exit 1; }

# 이미 켜져 있으면 브라우저만 연다
if curl -s -m 2 "$URL" >/dev/null; then
  echo "Bio-GPT가 이미 켜져 있습니다. 브라우저를 엽니다."; open "$URL"; exit 0
fi

step "1/5 Ollama(로컬 AI) 켜기"
command -v ollama >/dev/null || fail "ollama가 없습니다. https://ollama.com 에서 설치하세요."
if ! curl -s -m 2 localhost:11434/api/tags >/dev/null; then
  nohup ollama serve >/tmp/biogpt-ollama.log 2>&1 &
  for _ in $(seq 1 30); do curl -s -m 1 localhost:11434/api/tags >/dev/null && break; sleep 1; done
fi
curl -s -m 2 localhost:11434/api/tags >/dev/null || fail "Ollama가 켜지지 않습니다. /tmp/biogpt-ollama.log를 확인하세요."
echo "  켜짐"

step "2/5 AI 모델 확인 (처음 한 번만 받음)"
ollama list | grep -q "^bio-qwen3" || { ollama pull qwen3:8b && ollama create bio-qwen3 -f Modelfile.qwen3-bio; } || fail "bio-qwen3 준비 실패"
ollama list | grep -q "^bge-m3" || ollama pull bge-m3 || fail "bge-m3 받기 실패"
echo "  답변 모델 bio-qwen3, 임베딩 모델 bge-m3 준비됨"

step "3/5 라이브러리 확인"
command -v uv >/dev/null || fail "uv가 없습니다. curl -LsSf https://astral.sh/uv/install.sh | sh 로 설치하세요."
uv sync -q || fail "라이브러리 설치 실패"
echo "  준비됨"

step "4/5 모델 예열 (첫 질문이 오래 걸리지 않게 미리 메모리에 올림)"
curl -s -m 180 localhost:11434/api/generate -d '{"model":"bio-qwen3","prompt":"hi","stream":false,"keep_alive":"2h","options":{"num_predict":1}}' >/dev/null
curl -s -m 60 localhost:11434/api/embed -d '{"model":"bge-m3","input":"warm up","keep_alive":"2h"}' >/dev/null
echo "  완료"

step "5/5 Bio-GPT 웹 화면 열기 → $URL"
echo "  (끌 때: 이 창에서 Ctrl+C)"
# headless: Streamlit 첫 실행 때 터미널에서 이메일을 묻고 멈추는 것을 막는다. 브라우저는 준비되면 직접 연다.
( for _ in $(seq 1 60); do curl -s -m 1 "$URL/_stcore/health" | grep -q ok && { open "$URL"; break; }; sleep 1; done ) &
BIOGPT_SAMPLE=1 uv run streamlit run app.py --server.headless true --browser.gatherUsageStats false
