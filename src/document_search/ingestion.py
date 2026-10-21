"""Document parsing and sentence-aware chunking."""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import PurePath

from document_search.core import clean_text

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".pdf"}

# A boundary is a paragraph break, or whitespace after terminal punctuation followed
# by something that looks like the start of a new sentence (capital, digit, quote).
_SENTENCE_BOUNDARY = re.compile(r"\n|(?<=[.!?…]) (?=[\"'«(\[]?[A-ZА-ЯЁ0-9])")
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n")


class IngestionError(ValueError):
    """Raised when a document cannot be parsed into usable text.

    ``code`` is a stable reason the API translates for the user: unsupported_type,
    empty_file, bad_encoding, corrupted_pdf or no_text.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Chunk:
    """A piece of a document plus the metadata needed to cite it."""

    doc_id: str
    filename: str
    page: int | None  # 1-based page for PDFs, None for plain text
    chunk_index: int  # position of the chunk within its document
    char_start: int  # offsets into the cleaned page text
    char_end: int
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ParsedDocument:
    pages: list[tuple[int | None, str]]  # (page_number, raw_text) of readable pages
    skipped_pages: list[int] = field(default_factory=list)  # 1-based, unreadable PDF pages


@dataclass
class IngestResult:
    chunks: list[Chunk]
    skipped_pages: list[int] = field(default_factory=list)


def compute_doc_id(data: bytes) -> str:
    """Content hash, so re-uploading the same file maps to the same document."""
    return hashlib.sha256(data).hexdigest()[:16]


def file_extension(filename: str) -> str:
    return PurePath(filename).suffix.lower()


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1251"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise IngestionError("bad_encoding", "Text file is not valid UTF-8 or Windows-1251.")


# Glyphs PDF extractors emit when a font has no usable Unicode mapping.
_GARBAGE_CHARS = {"�", "■", "□", "▪", "▫"}
_MIN_LETTERS = 10


def _is_garbage_char(ch: str) -> bool:
    if ch in _GARBAGE_CHARS:
        return True
    category = unicodedata.category(ch)
    # Control (except whitespace), private-use and unassigned code points.
    return (category == "Cc" and ch not in "\n\r\t") or category in {"Co", "Cn"}


def is_readable_text(text: str) -> bool:
    """Heuristic: does extracted page text look like real language?

    Broken font mappings show up as replacement characters and squares, or as one
    glyph repeated for every letter (e.g. Cyrillic extracted as "IIIII IIII").
    """
    visible = [ch for ch in text if not ch.isspace()]
    if sum(ch.isalpha() for ch in visible) < _MIN_LETTERS:
        return False
    if sum(_is_garbage_char(ch) for ch in visible) / len(visible) > 0.05:
        return False
    words = re.findall(r"[^\W\d_]{3,}", text)
    repeated = sum(len(set(w)) == 1 for w in words)
    return not words or repeated / len(words) <= 0.3


def clean_unicode(text: str) -> str:
    """NFKC-normalize (ligatures, full-width forms) and drop soft hyphens and control chars."""
    text = unicodedata.normalize("NFKC", text).replace("­", "")
    return "".join(" " if _is_garbage_char(ch) else ch for ch in text)


def _parse_pdf(data: bytes) -> ParsedDocument:
    import pymupdf

    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except (pymupdf.FileDataError, RuntimeError, ValueError) as exc:
        raise IngestionError("corrupted_pdf", "PDF file is corrupted or cannot be read.") from exc
    pages: list[tuple[int | None, str]] = []
    skipped: list[int] = []
    with doc:
        if doc.needs_pass:
            raise IngestionError("corrupted_pdf", "PDF file is password-protected.")
        for number, page in enumerate(doc, 1):
            text = page.get_text("text")
            if is_readable_text(text):
                pages.append((number, clean_unicode(text)))
            elif text.strip() or page.get_images():
                skipped.append(number)  # has content we could not read (scan or broken font)
    return ParsedDocument(pages, skipped)


def parse_pages(data: bytes, filename: str) -> ParsedDocument:
    """Extract readable pages from a .txt or .pdf file."""
    ext = file_extension(filename)
    if ext not in SUPPORTED_EXTENSIONS:
        raise IngestionError("unsupported_type", f"Unsupported file type '{ext}'.")
    if not data:
        raise IngestionError("empty_file", "File is empty.")
    if ext == ".txt":
        return ParsedDocument([(None, clean_unicode(_decode_text(data)))])
    return _parse_pdf(data)


def normalize_text(raw: str) -> str:
    """Clean each paragraph and join paragraphs with a single newline.

    Blank lines mark paragraph breaks; single line breaks (hard wraps, PDF lines)
    are collapsed into spaces.
    """
    paragraphs = (clean_text(p) for p in _PARAGRAPH_BREAK.split(raw.replace("\r\n", "\n")))
    return "\n".join(p for p in paragraphs if p)


def split_sentence_spans(text: str) -> list[tuple[int, int]]:
    """Split text into sentences, returning (start, end) offsets."""
    spans = []
    start = 0
    for match in _SENTENCE_BOUNDARY.finditer(text):
        spans.append((start, match.start()))
        start = match.end()
    if start < len(text):
        spans.append((start, len(text)))
    return [(s, e) for s, e in spans if s < e]


def _split_long_span(text: str, start: int, end: int, max_chars: int) -> list[tuple[int, int]]:
    """Break a single over-long sentence on word boundaries."""
    pieces = []
    while end - start > max_chars:
        cut = text.rfind(" ", start, start + max_chars)
        if cut <= start:
            cut = start + max_chars
        pieces.append((start, cut))
        start = cut + 1 if text[cut : cut + 1] == " " else cut
    pieces.append((start, end))
    return pieces


def chunk_text(text: str, max_chars: int = 500, overlap_chars: int = 100) -> list[tuple[int, int]]:
    """Group whole sentences into chunks of at most ``max_chars`` characters.

    Consecutive chunks share trailing sentences totalling at most
    ``overlap_chars`` characters, so context is not lost at chunk edges.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if not 0 <= overlap_chars < max_chars:
        raise ValueError("overlap_chars must be in [0, max_chars)")

    spans = []
    for start, end in split_sentence_spans(text):
        spans.extend(_split_long_span(text, start, end, max_chars))

    chunks = []
    i, n = 0, len(spans)
    while i < n:
        j = i
        while j + 1 < n and spans[j + 1][1] - spans[i][0] <= max_chars:
            j += 1
        chunks.append((spans[i][0], spans[j][1]))
        if j == n - 1:
            break
        # Step back over trailing sentences to form the overlap, always moving forward.
        k = j + 1
        while k - 1 > i and spans[j][1] - spans[k - 1][0] <= overlap_chars:
            k -= 1
        i = k
    return chunks


def ingest_document(
    data: bytes,
    filename: str,
    *,
    max_chars: int = 500,
    overlap_chars: int = 100,
) -> IngestResult:
    """Parse, clean and chunk a document. Raises IngestionError on bad input."""
    doc_id = compute_doc_id(data)
    parsed = parse_pages(data, filename)
    chunks: list[Chunk] = []
    for page, raw in parsed.pages:
        text = normalize_text(raw)
        for start, end in chunk_text(text, max_chars, overlap_chars):
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    filename=filename,
                    page=page,
                    chunk_index=len(chunks),
                    char_start=start,
                    char_end=end,
                    text=text[start:end],
                )
            )
    if not chunks:
        raise IngestionError(
            "no_text", "No extractable text found (scanned PDFs are not supported)."
        )
    logger.info(
        "Ingested %s: %d chunks, %d skipped pages", filename, len(chunks), len(parsed.skipped_pages)
    )
    return IngestResult(chunks, parsed.skipped_pages)
