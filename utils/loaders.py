import re
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(text: str, metadata: dict, chunk_size: int = 512, chunk_overlap: int = 64) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    return [{"content": chunk, "metadata": dict(metadata)} for chunk in chunks if chunk.strip()]


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x20-\x7E\n]", "", text)
    return text.strip()


def build_document(
    content: str,
    company: str,
    source: str,
    quarter: Optional[str] = None,
    date: Optional[str] = None,
    url: Optional[str] = None,
) -> dict:
    return {
        "content": clean_text(content),
        "metadata": {
            "company": company,
            "source": source,
            "quarter": quarter or "unknown",
            "date": date or "unknown",
            "url": url or "",
        },
    }
