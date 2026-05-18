from __future__ import annotations

import re
from typing import Any


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", str(text)).strip()
    if not text:
        return []
    pieces = re.split(r"(?<=[.!?])\s+", text)
    return [piece.strip() for piece in pieces if piece.strip()]


def _normalize_rows(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    if not rows:
        return rows
    values = [float(row.get(key, 0.0)) for row in rows]
    lo = min(values)
    hi = max(values)
    span = hi - lo
    for row, value in zip(rows, values):
        row[f"{key}_norm"] = 1.0 if span == 0 else (value - lo) / span
    return rows


def evaluate_retrieval(
    question_embedding: Any,
    retrieved_chunks: list[dict[str, Any]],
    embedding_model: Any,
    min_correct_score: float,
    min_ambiguous_score: float,
) -> dict[str, Any]:
    """CRAG retrieval evaluator.

    The CRAG paper uses a lightweight evaluator to assign a confidence degree and
    trigger correct/ambiguous/incorrect actions. This local implementation keeps
    that decision structure while using the same sentence embedding model already
    used by the Kiraffe-style retriever.
    """
    import numpy as np

    if not retrieved_chunks:
        return {"state": "incorrect", "confidence": 0.0, "chunk_scores": []}

    texts = [str(chunk.get("text", "")) for chunk in retrieved_chunks]
    chunk_embeddings = embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    norms = np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    chunk_embeddings = chunk_embeddings / norms
    scores = chunk_embeddings @ question_embedding

    chunk_scores = []
    for chunk, score in zip(retrieved_chunks, scores):
        compact = {
            "rank": chunk.get("rank"),
            "score": chunk.get("score"),
            "crag_score": float(score),
            "topic_id": chunk.get("topic_id"),
            "article_id": chunk.get("article_id"),
            "title": chunk.get("title"),
            "chunk_id": chunk.get("chunk_id"),
        }
        chunk_scores.append(compact)

    confidence = float(max(scores)) if len(scores) else 0.0
    mean_top = float(np.mean(np.sort(scores)[-min(3, len(scores)) :])) if len(scores) else 0.0
    decision_score = max(confidence, mean_top)
    if decision_score >= min_correct_score:
        state = "correct"
    elif decision_score >= min_ambiguous_score:
        state = "ambiguous"
    else:
        state = "incorrect"
    return {"state": state, "confidence": decision_score, "chunk_scores": chunk_scores}


def refine_chunks(
    question_embedding: Any,
    retrieved_chunks: list[dict[str, Any]],
    embedding_model: Any,
    top_sentences: int,
    min_sentence_score: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Decompose retrieved chunks into sentences and recompose relevant evidence."""
    import numpy as np

    sentence_rows = []
    for chunk in retrieved_chunks:
        for sent_index, sentence in enumerate(split_sentences(str(chunk.get("text", "")))):
            sentence_rows.append({"chunk": chunk, "sent_index": sent_index, "sentence": sentence})
    if not sentence_rows:
        return retrieved_chunks, []

    sentence_embeddings = embedding_model.encode(
        [row["sentence"] for row in sentence_rows],
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)
    norms = np.linalg.norm(sentence_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    sentence_embeddings = sentence_embeddings / norms
    scores = sentence_embeddings @ question_embedding
    for row, score in zip(sentence_rows, scores):
        row["score"] = float(score)
    sentence_rows.sort(key=lambda row: row["score"], reverse=True)
    selected = [row for row in sentence_rows if row["score"] >= min_sentence_score]
    if not selected:
        selected = sentence_rows[: max(1, min(top_sentences, len(sentence_rows)))]
    else:
        selected = selected[:top_sentences]

    recomposed = []
    for rank, row in enumerate(selected, 1):
        chunk = row["chunk"]
        recomposed.append(
            {
                **chunk,
                "rank": rank,
                "text": row["sentence"],
                "source_rank": chunk.get("rank"),
                "sentence_index": row["sent_index"],
                "crag_sentence_score": row["score"],
            }
        )
    evidence = [
        {
            "rank": row.get("source_rank", row.get("rank")),
            "title": row.get("title"),
            "topic_id": row.get("topic_id"),
            "sentence_index": row.get("sentence_index"),
            "score": row.get("crag_sentence_score"),
            "text": row.get("text"),
        }
        for row in recomposed
    ]
    return recomposed, evidence
