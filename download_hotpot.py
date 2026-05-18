#!/usr/bin/env python3
"""Download HotpotQA from HuggingFace and save HotpotQA-format JSON.

Uses exactly these dataset configs:

    load_dataset("hotpotqa/hotpot_qa", "fullwiki")
    load_dataset("hotpotqa/hotpot_qa", "distractor")

The HuggingFace validation split is converted to the original HotpotQA JSON
shape expected by the naive RAG and evaluator.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

from datasets import load_dataset


OUT_FILES = {
    "fullwiki": "hotpot_dev_fullwiki_v1.json",
    "distractor": "hotpot_dev_distractor_v1.json",
}


def convert_example(item: dict[str, Any]) -> dict[str, Any]:
    context = []
    context_data = item.get("context", {})
    for title, sentences in zip(context_data.get("title", []), context_data.get("sentences", [])):
        context.append([title, sentences])

    supporting_facts = []
    support_data = item.get("supporting_facts", {})
    for title, sent_id in zip(support_data.get("title", []), support_data.get("sent_id", [])):
        supporting_facts.append([title, int(sent_id)])

    return {
        "_id": str(item.get("id", "")),
        "question": str(item.get("question", "")),
        "answer": str(item.get("answer", "")),
        "type": str(item.get("type", "")),
        "level": str(item.get("level", "")),
        "context": context,
        "supporting_facts": supporting_facts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Download HotpotQA from HuggingFace.")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--splits", nargs="+", default=["fullwiki", "distractor"], choices=sorted(OUT_FILES))
    parser.add_argument("--hf-split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for config in args.splits:
        out_path = out_dir / OUT_FILES[config]
        if out_path.exists() and not args.force:
            print(f"exists: {out_path}")
            continue

        print(f"loading HuggingFace hotpotqa/hotpot_qa config={config} split={args.hf_split}", flush=True)
        ds = load_dataset("hotpotqa/hotpot_qa", config, split=args.hf_split)
        examples = []
        for idx, item in enumerate(ds, start=1):
            examples.append(convert_example(item))
            if idx % 1000 == 0:
                print(f"converted {idx}/{len(ds)} {config} examples", flush=True)

        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(examples, handle, ensure_ascii=False)
        print(f"wrote: {out_path} ({len(examples)} examples)", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
