import re
import zlib

import numpy as np
import pytest

from document_search.retrieval import BaseEmbedder
from document_search.retrieval.embeddings import l2_normalize


class HashingEmbedder(BaseEmbedder):
    """Deterministic bag-of-words embedder: no model download, no network."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for word in re.findall(r"\w+", text.lower()):
            vec[zlib.crc32(word.encode()) % self.dim] += 1.0
        return vec

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return l2_normalize(np.stack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return l2_normalize(self._vector(text))


@pytest.fixture
def embedder():
    return HashingEmbedder()


def build_pdf(pages: list[str]) -> bytes:
    """Build a minimal valid PDF with one line of ASCII text per page."""
    page_ids = [4 + 2 * i for i in range(len(pages))]
    kids = " ".join(f"{p} 0 R" for p in page_ids)
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for page_id, text in zip(page_ids, pages):
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {page_id + 1} 0 R >>".encode()
        )
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")

    out = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n"
    ).encode()
    return out


@pytest.fixture
def make_pdf():
    return build_pdf


def build_unicode_pdf(pages: list[str | None]) -> bytes:
    """Build a PDF with real Unicode text (e.g. Cyrillic) per page; None makes a blank page."""
    import pymupdf

    doc = pymupdf.open()
    font = pymupdf.Font("helv")  # PyMuPDF's built-in Helvetica covers Cyrillic
    for text in pages:
        page = doc.new_page()
        if text:
            writer = pymupdf.TextWriter(page.rect)
            y = 72
            for line in text.split("\n"):
                writer.append((72, y), line, font=font, fontsize=11)
                y += 16
            writer.write_text(page)
    return doc.tobytes()


@pytest.fixture
def make_unicode_pdf():
    return build_unicode_pdf
