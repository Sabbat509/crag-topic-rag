#!/usr/bin/env python3
"""Build reference-style doc_data.pkl from HotpotQA Wikipedia abstracts."""

from __future__ import annotations

import argparse
import bz2
import json
import pickle
import tarfile
import time
from pathlib import Path


def iter_docs(archive: Path):
    with tarfile.open(archive, "r:bz2") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".bz2"):
                continue
            fileobj = tar.extractfile(member)
            if fileobj is None:
                continue
            with bz2.open(fileobj, "rt", encoding="utf-8") as inner:
                for line in inner:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    text = " ".join(str(sent) for sent in item.get("text", []))
                    if text:
                        yield {
                            "id": item.get("id"),
                            "title": item.get("title", "Unknown"),
                            "text": text,
                            "source_file": member.name,
                            "url": item.get("url"),
                        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", default="data/wiki/hotpot_wiki_abstracts.tar.bz2")
    parser.add_argument("--out", default="result/intermediate/doc_data.pkl")
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--progress-every", type=int, default=250000)
    args = parser.parse_args()

    archive = Path(args.archive)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    docs = []
    started = time.time()
    for idx, doc in enumerate(iter_docs(archive), 1):
        docs.append(doc)
        if idx % args.progress_every == 0:
            print(f"read {idx:,} docs in {(time.time()-started)/60:.1f} min", flush=True)
        if args.max_documents is not None and idx >= args.max_documents:
            break
    with out.open("wb") as f:
        pickle.dump(docs, f)
    print(f"wrote: {out}")
    print(f"documents: {len(docs):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
