"""Download the public-domain King James Version Gospel of Mark as plain text."""
from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
CORPUS_DIR = ROOT / "corpus"
OUTPUT_FILE = CORPUS_DIR / "mark.txt"

# Try the per-book JSON first; fall back to a complete KJV JSON file.
SOURCE_URLS = [
    "https://raw.githubusercontent.com/nolanbaxter/kjv-bible/main/mark.json",
    "https://raw.githubusercontent.com/nolanbaxter/kjv-bible/master/mark.json",
    "https://raw.githubusercontent.com/thiagobodruk/bible/master/json/en_kjv.json",
]


def get_mark_book(data: Any) -> dict:
    """Return a Mark book record from common KJV JSON layouts."""
    candidates = []
    if isinstance(data, dict):
        if str(data.get("book", data.get("name", ""))).strip().lower() == "mark":
            return data
        for key in ("books", "data", "bible"):
            if isinstance(data.get(key), list):
                candidates.extend(data[key])
    elif isinstance(data, list):
        candidates.extend(data)

    for book in candidates:
        if isinstance(book, dict) and str(
            book.get("book", book.get("name", book.get("book_name", "")))
        ).strip().lower() == "mark":
            return book
    raise ValueError("Could not find the Gospel of Mark in the downloaded JSON.")


def iter_verses(book: dict):
    chapters = book.get("chapters")
    if not chapters:
        raise ValueError("The Mark JSON does not contain a 'chapters' field.")

    if isinstance(chapters, dict):
        chapter_items = sorted(chapters.items(), key=lambda item: int(item[0]))
        for chapter_num, verses in chapter_items:
            if isinstance(verses, dict):
                verse_items = sorted(verses.items(), key=lambda item: int(item[0]))
            elif isinstance(verses, list):
                verse_items = [(str(i), verse) for i, verse in enumerate(verses, 1)]
            else:
                continue
            for verse_num, text in verse_items:
                if isinstance(text, str) and text.strip():
                    yield int(chapter_num), int(verse_num), text.strip()

    elif isinstance(chapters, list):
        for chapter_index, verses in enumerate(chapters, start=1):
            if isinstance(verses, dict):
                verse_items = sorted(verses.items(), key=lambda item: int(item[0]))
            elif isinstance(verses, list):
                verse_items = [(str(i), verse) for i, verse in enumerate(verses, 1)]
            else:
                continue
            for verse_num, text in verse_items:
                if isinstance(text, str) and text.strip():
                    yield chapter_index, int(verse_num), text.strip()
    else:
        raise ValueError("Unrecognized chapter layout in downloaded JSON.")


def download_json(url: str) -> Any:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Lab2-RAG educational project/1.0"},
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    last_error = None
    for url in SOURCE_URLS:
        try:
            print(f"Downloading corpus source: {url}")
            data = download_json(url)
            book = get_mark_book(data)
            lines = []
            for chapter, verse, text in iter_verses(book):
                # Remove display-only markers used in some KJV JSON files.
                text = re.sub(r"[\[\]{}]", "", text).strip()
                text = re.sub(r"\s+", " ", text)
                lines.append(f"Mark {chapter}:{verse} | {text}")
            if len(lines) < 600:
                raise ValueError(f"Expected hundreds of verses; received only {len(lines)}.")
            OUTPUT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"Saved {len(lines)} verses to: {OUTPUT_FILE}")
            print("Source text: King James Version (public domain in the United States).")
            return
        except Exception as exc:
            last_error = exc
            print(f"Could not use this source: {exc}")

    raise RuntimeError(
        "Could not download the Gospel of Mark. Check your internet connection, "
        "then run 'python download_corpus.py' again. "
        f"Last error: {last_error}"
    )


if __name__ == "__main__":
    main()
