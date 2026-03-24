import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face dataset on a machine with internet and export it to local txt/jsonl files."
    )
    parser.add_argument("--dataset_name", type=str, required=True, help="HF dataset name, e.g. rojagtap/bookcorpus")
    parser.add_argument("--split", type=str, default="train", help="Dataset split, e.g. train")
    parser.add_argument("--text_column", type=str, default="text", help="Name of the text column")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for exported files")
    parser.add_argument("--cache_dir", type=str, default=None, help="Optional HF cache dir")
    parser.add_argument("--max_texts", type=int, default=None, help="Optional max number of texts to export")
    parser.add_argument("--file_format", type=str, default="txt", choices=["txt", "jsonl"], help="Export format")
    parser.add_argument("--filename", type=str, default=None, help="Optional output filename")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset_name, split=args.split, cache_dir=args.cache_dir)

    if args.max_texts is not None:
        ds = ds.select(range(min(args.max_texts, len(ds))))

    if args.text_column not in ds.column_names:
        raise ValueError(
            f"text_column={args.text_column!r} not found. Available columns: {ds.column_names}"
        )

    filename = args.filename
    if filename is None:
        safe_name = args.dataset_name.replace("/", "__")
        ext = "txt" if args.file_format == "txt" else "jsonl"
        filename = f"{safe_name}_{args.split}.{ext}"

    output_path = output_dir / filename

    count = 0
    if args.file_format == "txt":
        with open(output_path, "w", encoding="utf-8") as f:
            for row in ds:
                text = row.get(args.text_column, "")
                if text is None:
                    continue
                text = str(text).strip()
                if not text:
                    continue
                f.write(text + "\n")
                count += 1
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            for row in ds:
                text = row.get(args.text_column, "")
                if text is None:
                    continue
                text = str(text).strip()
                if not text:
                    continue
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                count += 1

    print(f"Saved {count} texts to: {output_path}")
    print("Next step: upload this output_dir to your server with rsync/scp/Xftp.")


if __name__ == "__main__":
    main()
