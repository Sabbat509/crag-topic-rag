#!/usr/bin/env python3
"""Build Kiraffe-style BERTopic topic partition artifacts for HotpotQA.

Outputs:
- topics_<N>.npy
- valid_indices_<N>.pkl
- topic_info_<N>.csv
- topic_article_map.jsonl/.csv/manifest.json

This mirrors the BERTopic-for-RAG flow: topic assignments are produced at the
article/document level and then converted into an article-to-topic map for a
later topic-aware chunk index.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from naive_rag.index_io import load_pickle, save_json, save_pickle, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BERTopic topic partition artifacts for CRAG topic RAG.")
    parser.add_argument("--doc-data", type=Path, default=Path("result/intermediate/doc_data.pkl"))
    parser.add_argument("--output-dir", type=Path, default=Path("result/crag_topic_rag/topic_partition"))
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--nr-topics", type=int, default=40)
    parser.add_argument("--min-topic-size", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device")
    parser.add_argument("--umap-n-neighbors", type=int, default=15)
    parser.add_argument("--umap-n-components", type=int, default=2)
    parser.add_argument("--umap-min-dist", type=float, default=0.0)
    parser.add_argument("--hdbscan-min-samples", type=int, default=10)
    parser.add_argument("--max-df", type=float, default=0.85)
    parser.add_argument("--max-features", type=int, default=10000)
    parser.add_argument("--top-n-words", type=int, default=12)
    parser.add_argument("--max-documents", type=int, help="Optional smoke/debug cap. Omit for full corpus.")
    parser.add_argument("--min-words", type=int, default=15)
    return parser.parse_args()


def valid_documents(doc_data: list[dict[str, Any]], max_documents: int | None, min_words: int) -> tuple[list[str], list[int]]:
    docs = []
    valid_indices = []
    for raw_index, item in enumerate(doc_data):
        text = str(item.get("text", "")).strip()
        if len(text.split()) < min_words:
            continue
        docs.append(text)
        valid_indices.append(raw_index)
        if max_documents is not None and len(docs) >= max_documents:
            break
    if not docs:
        raise ValueError("No valid documents were found for topic modeling.")
    return docs, valid_indices


def topic_info_to_csv(topic_model: Any, path: Path) -> None:
    info = topic_model.get_topic_info()
    path.parent.mkdir(parents=True, exist_ok=True)
    info.to_csv(path, index=False)


def build_rows(topics: list[int], valid_indices: list[int], doc_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for topic_row, (topic_id, raw_doc_index) in enumerate(zip(topics, valid_indices)):
        item = doc_data[int(raw_doc_index)]
        text = item.get("text", "")
        rows.append(
            {
                "topic_row": int(topic_row),
                "topic_id": int(topic_id),
                "raw_doc_index": int(raw_doc_index),
                "article_id": str(item.get("id", raw_doc_index)),
                "title": item.get("title", "Unknown"),
                "source_file": item.get("source_file"),
                "text_length": len(text) if isinstance(text, str) else 0,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["topic_row", "topic_id", "raw_doc_index", "article_id", "title", "source_file", "text_length"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    import numpy as np
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    doc_data = load_pickle(args.doc_data)
    docs, valid_indices = valid_documents(doc_data, args.max_documents, args.min_words)
    print(f"Documents available: {len(doc_data):,}")
    print(f"Documents used for BERTopic: {len(docs):,}")

    model_kwargs = {"device": args.device} if args.device else {}
    embedding_model = SentenceTransformer(args.embedding_model, **model_kwargs)
    embeddings = embedding_model.encode(
        docs,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    umap_model = UMAP(
        n_neighbors=args.umap_n_neighbors,
        n_components=args.umap_n_components,
        min_dist=args.umap_min_dist,
        metric="cosine",
        random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=args.min_topic_size,
        min_samples=args.hdbscan_min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=False,
    )
    vectorizer_model = CountVectorizer(
        stop_words="english",
        max_df=args.max_df,
        max_features=args.max_features,
        ngram_range=(1, 2),
    )
    topic_model = BERTopic(
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        nr_topics=args.nr_topics,
        top_n_words=args.top_n_words,
        min_topic_size=args.min_topic_size,
        calculate_probabilities=False,
        verbose=True,
    )
    topics, _ = topic_model.fit_transform(docs, embeddings=np.asarray(embeddings))

    stem = str(args.nr_topics)
    topics_path = args.output_dir / f"topics_{stem}.npy"
    valid_indices_path = args.output_dir / f"valid_indices_{stem}.pkl"
    topic_info_path = args.output_dir / f"topic_info_{stem}.csv"
    map_jsonl_path = args.output_dir / "topic_article_map.jsonl"
    map_csv_path = args.output_dir / "topic_article_map.csv"
    manifest_path = args.output_dir / "topic_partition_manifest.json"

    np.save(topics_path, np.asarray(topics, dtype=np.int32))
    save_pickle(valid_indices_path, [int(index) for index in valid_indices])
    topic_info_to_csv(topic_model, topic_info_path)
    rows = build_rows([int(topic) for topic in topics], valid_indices, doc_data)
    write_jsonl(map_jsonl_path, rows)
    write_csv(map_csv_path, rows)
    topic_counts = Counter(row["topic_id"] for row in rows)
    save_json(
        manifest_path,
        {
            "created_at": timestamp,
            "doc_data_path": str(args.doc_data.resolve()),
            "embedding_model": args.embedding_model,
            "nr_topics": args.nr_topics,
            "min_topic_size": args.min_topic_size,
            "hdbscan_min_samples": args.hdbscan_min_samples,
            "umap_n_neighbors": args.umap_n_neighbors,
            "umap_n_components": args.umap_n_components,
            "umap_min_dist": args.umap_min_dist,
            "max_df": args.max_df,
            "max_features": args.max_features,
            "top_n_words": args.top_n_words,
            "document_count_available": len(doc_data),
            "document_count_topic_modeled": len(docs),
            "topic_count_including_outlier": len(topic_counts),
            "outlier_count": topic_counts.get(-1, 0),
            "topics_path": str(topics_path.resolve()),
            "valid_indices_path": str(valid_indices_path.resolve()),
            "topic_info_path": str(topic_info_path.resolve()),
            "topic_article_map_jsonl": str(map_jsonl_path.resolve()),
            "topic_article_map_csv": str(map_csv_path.resolve()),
        },
    )
    print(f"Saved topics: {topics_path}")
    print(f"Saved topic info: {topic_info_path}")
    print(f"Saved topic article map: {map_jsonl_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
