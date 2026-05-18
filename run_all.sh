#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

.venv/bin/python download_hotpot.py --out-dir data --splits fullwiki

if [[ ! -f data/wiki/hotpot_wiki_abstracts.tar.bz2 ]]; then
  echo "Missing data/wiki/hotpot_wiki_abstracts.tar.bz2"
  echo "Download: https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2"
  exit 1
fi

mkdir -p result/intermediate result/base_index result/crag_topic_rag logs

if [[ ! -f result/intermediate/doc_data.pkl ]]; then
  .venv/bin/python build_reference_doc_data.py \
    --wiki-tar data/wiki/hotpot_wiki_abstracts.tar.bz2 \
    --out-file result/intermediate/doc_data.pkl \
    2>&1 | tee logs/build_reference_doc_data.log
fi

if [[ ! -f result/base_index/chunks.jsonl || ! -f result/base_index/embeddings.npy ]]; then
  .venv/bin/python build_reference_index.py \
    --doc-data result/intermediate/doc_data.pkl \
    --output-dir result/base_index \
    --embedding-model all-MiniLM-L6-v2 \
    --batch-size 32 \
    --chunk-size-words 400 \
    --chunk-overlap-words 80 \
    --min-chunk-words 40 \
    --device cuda:0 \
    2>&1 | tee logs/build_reference_index.log
fi

if [[ ! -f result/crag_topic_rag/topic_partition/topic_article_map.jsonl ]]; then
  .venv/bin/python build_topic_partition.py \
    --doc-data result/intermediate/doc_data.pkl \
    --output-dir result/crag_topic_rag/topic_partition \
    --nr-topics 40 \
    --device cuda:0 \
    2>&1 | tee logs/build_topic_partition.log
fi

if [[ ! -f result/crag_topic_rag/topic_index/chunks.jsonl ]]; then
  .venv/bin/python build_crag_topic_index.py \
    --base-index-dir result/base_index \
    --topic-map result/crag_topic_rag/topic_partition/topic_article_map.jsonl \
    --topic-info result/crag_topic_rag/topic_partition/topic_info_40.csv \
    --output-dir result/crag_topic_rag/topic_index \
    2>&1 | tee logs/build_crag_topic_index.log
fi

.venv/bin/python evaluate_crag_topic_rag.py \
  --index-dir result/crag_topic_rag/topic_index \
  --topic-info result/crag_topic_rag/topic_partition/topic_info_40.csv \
  --data-file data/hotpot_dev_fullwiki_v1.json \
  --output-dir result/crag_topic_rag/hotpotqa_eval/reference_crag_topic_fullwiki \
  --device cuda:0 \
  --resume \
  --progress-every 25 \
  --no-save-prompts \
  2>&1 | tee logs/reference_crag_topic_fullwiki_eval.log
