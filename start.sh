#!/bin/bash

# 修正：在这里直接填入完整的 API 地址 (包含端口和 /v1)
LLM_URL="http://172.17.0.1:8091/v1"
OUTPUT_BASE="./results/arxiv_review"

if [ -z "$1" ]; then
  echo "用法: bash start.sh \"<论文主题>\" [ArXiv分类] [关键词]"
  exit 1
fi

TOPIC="$1"
CATEGORY="${2:-cs.SE}"
KEYWORD="${3:-}"

export OPENAI_TIMEOUT=600 

echo "🚀 第一步：生成内容..."
# 这里直接使用修正后的 LLM_URL
python run_storm_arxiv.py \
  --llm-url "$LLM_URL" \
  --output-dir "$OUTPUT_BASE" \
  --topic "$TOPIC" \
  --category "$CATEGORY" \
  --keyword "$KEYWORD"

if [ $? -ne 0 ]; then echo "❌ 生成失败"; exit 1; fi

DIR_NAME=$(echo "$TOPIC" | sed 's/ /_/g' | sed 's/\//_/g')
RESULT_DIR="$OUTPUT_BASE/$DIR_NAME"

echo "🚀 第二步：生成 LaTeX..."
# 这里也不需要再手动拼接端口了
python storm_final_processing.py \
  --dir "$RESULT_DIR" \
  --llm-url "$LLM_URL"

echo "🎉 全部完成！结果目录: $RESULT_DIR"