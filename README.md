# CRAG Topic RAG for HotpotQA

CRAG-style corrective retrieval for HotpotQA using the same BERTopic topic partitioning pattern as [`Kiraffe1206/BERTopic-for-RAG`](https://github.com/Kiraffe1206/BERTopic-for-RAG).

This repository is for the **CRAG + topic partitioning** experiment only. It does not include the Naive RAG evaluator or Naive RAG result artifacts.


## Completed Result

Evaluation split: `hotpotqa/hotpot_qa`, `fullwiki`, validation.

| Method | EM | Acc | F1 | G-Sem | Tok |
|---|---:|---:|---:|---:|---:|
| CRAG + Topic RAG | 10.61 | 13.75 | 13.50 | 32.31 | 1.30 |

LaTeX row:

```latex
CRAG + Topic RAG & 10.61 & 13.75 & 13.50 & 32.31 & 1.30
```

Final CRAG retrieval labels:

```text
correct: 6520
ambiguous: 684
incorrect: 201
```

The tracked summary file is `docs/results/summary.json`. Full `records.jsonl` and `predictions.csv` outputs are generated locally and ignored by git.

## Pipeline

```text
HotpotQA fullwiki question
  -> Gemma extracts retrieval knowledge points
  -> rank BERTopic topic profiles with all-MiniLM-L6-v2
  -> retrieve top-k chunks from selected topics only
  -> CRAG retrieval evaluator labels retrieval as correct/ambiguous/incorrect
  -> decompose chunks into sentences and recompose relevant evidence
  -> if retrieval is weak, add corrective full-index retrieval
  -> Gemma generates the shortest answer
  -> compute EM, Acc, F1, G-Sem, Tok
```

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

Build Kiraffe-style BERTopic topic partition artifacts:

```bash
.venv/bin/python build_topic_partition.py \
  --doc-data result/intermediate/doc_data.pkl \
  --output-dir result/crag_topic_rag/topic_partition \
  --nr-topics 40 \
  --backend auto \
  --device cuda:0
```

Attach topic IDs to chunks:

```bash
.venv/bin/python build_crag_topic_index.py \
  --base-index-dir result/base_index \
  --topic-map result/crag_topic_rag/topic_partition/topic_article_map.jsonl \
  --topic-info result/crag_topic_rag/topic_partition/topic_info_40.csv \
  --output-dir result/crag_topic_rag/topic_index
```

Run CRAG + topic RAG evaluation:

```bash
.venv/bin/python evaluate_crag_topic_rag.py \
  --index-dir result/crag_topic_rag/topic_index \
  --topic-info result/crag_topic_rag/topic_partition/topic_info_40.csv \
  --data-file data/hotpot_dev_fullwiki_v1.json \
  --output-dir result/crag_topic_rag/hotpotqa_eval/reference_crag_topic_fullwiki \
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
