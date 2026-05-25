# CRAG Topic RAG for HotpotQA

CRAG-style corrective retrieval for HotpotQA using the same BERTopic topic partitioning pattern as [`Kiraffe1206/BERTopic-for-RAG`](https://github.com/Kiraffe1206/BERTopic-for-RAG).

This repository is for the **CRAG + topic partitioning** experiment only. It does not include the Naive RAG evaluator or Naive RAG result artifacts.


## Completed Result

Evaluation split: `hotpotqa/hotpot_qa`, `fullwiki`, validation.

| Method | EM | Acc | F1 | G-Sem | Tok |
|---|---:|---:|---:|---:|---:|
| Topic-Partitioned CRAG | 19.86 | 26.20 | 25.89 | 45.95 | 1.55 |

LaTeX row:

```latex
Topic-Partitioned CRAG & 19.86 & 26.20 & 25.89 & 45.95 & 1.55
```

Final CRAG branch counts:

```text
correct: 1735
ambiguous: 3346
incorrect: 2324
```

The tracked summary file is `docs/results/summary.json`. Full `records.jsonl` and `predictions.csv` outputs are generated locally and ignored by git.

## Kiraffe 2026-05-22 Update

Kiraffe's updated BERTopic hyperparameter search no longer recommends the old fixed 40-topic HotpotQA partition. The recommended HotpotQA setting is:

```text
uc=5, un=30, hcs=20, hms=20, max_df=0.95, ngram=(1,2), nr_topics=None
```

The reported natural topic count is approximately 252 topics from repeated staged samples. This repo keeps the original completed 40-topic result above for traceability, but new reruns should use the `topic_partition_kiraffe_20260522` / `topic_index_kiraffe_20260522` commands below.

## Pipeline

```text
HotpotQA fullwiki question
  -> Gemma extracts retrieval knowledge points
  -> rank BERTopic topic profiles with all-MiniLM-L6-v2
  -> retrieve top-k chunks from selected topics only
  -> score query-passage pairs with a CRAG-style retrieval evaluator
  -> convert scores to CRAG flags: 2=correct, 1=ambiguous, 0=incorrect
  -> choose the CRAG branch:
       correct: use internal topic-retrieved knowledge
       ambiguous: combine internal knowledge + external fallback knowledge
       incorrect: use external fallback knowledge
  -> decompose/recompose knowledge strips following the CRAG repo modes
  -> Gemma generates the shortest answer
  -> compute EM, Acc, F1, G-Sem, Tok
```

## CRAG Adaptation Notes

This repository follows the public [`HuskyInSalt/CRAG`](https://github.com/HuskyInSalt/CRAG) control flow: retrieval evaluation, score-to-flag conversion, `correct` / `ambiguous` / `incorrect` branching, and decompose-then-recompose knowledge preparation.

Two adaptations are used for HotpotQA:

- Topic partitioning is added before retrieval, following the BERTopic-for-RAG setup.
- The official CRAG web-search branch is replaced with fallback retrieval over the fixed HotpotQA fullwiki corpus. This keeps the experiment reproducible and avoids external search API drift.

To run a web-search version closer to the original CRAG setup, a search provider key is needed, such as a `serper.dev` API key. The original CRAG repo also uses an OpenAI key during external knowledge preparation. Without those keys, this repository reports the reproducible fullwiki-fallback variant.

## Main Files

| Path | Purpose |
|---|---|
| `download_hotpot.py` | Downloads HotpotQA fullwiki validation data from HuggingFace. |
| `build_reference_doc_data.py` | Converts the HotpotQA Wikipedia abstracts dump into document data. |
| `build_reference_index.py` | Builds the full vector chunk index used for retrieval/fallback. |
| `build_topic_partition.py` | Builds Kiraffe-style BERTopic topic artifacts and article-to-topic map. |
| `build_crag_topic_index.py` | Attaches BERTopic `topic_id` values to the chunk index. |
| `evaluate_crag_topic_rag.py` | Runs CRAG + topic-partitioned HotpotQA evaluation. |
| `crag_topic_rag/` | CRAG evaluator, sentence refinement, and topic router. |
| `naive_rag/` | Shared helpers only: chunking, index I/O, and local LLM client. |

## Requirements

Install dependencies:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Serve Gemma locally with Ollama:

```bash
ollama serve
ollama pull gemma2:2b
```

Large generated data is ignored by git.

## Reproduce

Download HotpotQA:

```bash
.venv/bin/python download_hotpot.py --out-dir data --splits fullwiki
```

Download the official HotpotQA Wikipedia abstracts corpus to:

```text
data/wiki/hotpot_wiki_abstracts.tar.bz2
```

Build document data:

```bash
.venv/bin/python build_reference_doc_data.py \
  --wiki-tar data/wiki/hotpot_wiki_abstracts.tar.bz2 \
  --out-file result/intermediate/doc_data.pkl
```

Build the full vector chunk index:

```bash
.venv/bin/python build_reference_index.py \
  --doc-data result/intermediate/doc_data.pkl \
  --output-dir result/base_index \
  --embedding-model all-MiniLM-L6-v2 \
  --batch-size 32 \
  --chunk-size-words 400 \
  --chunk-overlap-words 80 \
  --min-chunk-words 40 \
  --device cuda:0
```

Build Kiraffe-style BERTopic topic partition artifacts using the 2026-05-22 recommended HotpotQA settings (`uc=5`, `un=30`, `hcs=20`, `hms=20`, `max_df=0.95`, `ngram=(1,2)`). Kiraffe reports about 252 natural topics from staged sampling; this command lets HDBSCAN keep the natural topic count with `nr_topics=None`:

```bash
.venv/bin/python build_topic_partition.py \
  --doc-data result/intermediate/doc_data.pkl \
  --output-dir result/crag_topic_rag/topic_partition_kiraffe_20260522 \
  --nr-topics none \
  --umap-n-components 5 \
  --umap-n-neighbors 30 \
  --min-topic-size 20 \
  --hdbscan-min-samples 20 \
  --max-df 0.95 \
  --ngram-range 1,2 \
  --min-df 2 \
  --backend auto \
  --device cuda:0
```

Attach topic IDs to chunks:

```bash
.venv/bin/python build_crag_topic_index.py \
  --base-index-dir result/base_index \
  --topic-map result/crag_topic_rag/topic_partition_kiraffe_20260522/topic_article_map.jsonl \
  --topic-info result/crag_topic_rag/topic_partition_kiraffe_20260522/topic_info_none.csv \
  --output-dir result/crag_topic_rag/topic_index_kiraffe_20260522
```

Run CRAG + topic RAG evaluation:

```bash
.venv/bin/python evaluate_crag_topic_rag.py \
  --index-dir result/crag_topic_rag/topic_index_kiraffe_20260522 \
  --topic-info result/crag_topic_rag/topic_partition_kiraffe_20260522/topic_info_none.csv \
  --data-file data/hotpot_dev_fullwiki_v1.json \
  --output-dir result/crag_topic_rag/hotpotqa_eval/reference_crag_topic_kiraffe_20260522_fullwiki \
  --device cuda:0 \
  --resume \
  --progress-every 25 \
  --no-save-prompts
```

Outputs:

```text
result/crag_topic_rag/hotpotqa_eval/reference_crag_topic_fullwiki/records.jsonl
result/crag_topic_rag/hotpotqa_eval/reference_crag_topic_fullwiki/predictions.csv
result/crag_topic_rag/hotpotqa_eval/reference_crag_topic_fullwiki/summary.json
```

## Metrics

The evaluator writes the same table metrics used for the HotpotQA experiments:

- `EM`: normalized exact match
- `Acc`: relaxed answer accuracy
- `F1`: token-level answer F1
- `G-Sem`: sentence-embedding semantic similarity between prediction and gold
- `Tok`: average normalized output tokens
