#!/usr/bin/env bash
# 코멘토 2주차 제출용 zip 만들기:  bash scripts/make_submission.sh 홍길동
#  → ../2주차_프로토타입_홍길동.zip  (git에 커밋된 파일만 담는다. .env·색인·가상환경은 자동 제외)
set -euo pipefail
NAME="${1:?사용법: bash scripts/make_submission.sh <이름>}"
cd "$(dirname "$0")/.."
OUT="../2주차_프로토타입_${NAME}.zip"
# 안쪽 폴더 이름은 영문으로 둔다. 한글 폴더명은 윈도우·macOS unzip에서 깨지거나 풀리지 않는다
git archive --format=zip --prefix="bio-gpt-prototype/" -o "$OUT" HEAD
echo "만듦: $(cd .. && pwd)/2주차_프로토타입_${NAME}.zip ($(du -h "$OUT" | cut -f1))"
