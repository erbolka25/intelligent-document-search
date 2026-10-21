"""Request and response models."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

from document_search.retrieval import SearchResult

QueryText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class LoginRequest(BaseModel):
    password: Annotated[str, StringConstraints(min_length=1, max_length=256)]


class OkResponse(BaseModel):
    ok: bool


class DeleteResponse(BaseModel):
    deleted: int


class SearchRequest(BaseModel):
    query: QueryText
    top_k: int = Field(default=5, ge=1, le=20)


class AskRequest(BaseModel):
    question: QueryText
    top_k: int = Field(default=5, ge=1, le=20)


class ChunkOut(BaseModel):
    rank: int = Field(description="Position in the results; also the [n] source number in answers")
    score: float
    doc_id: str
    filename: str
    page: int | None
    chunk_index: int
    char_start: int
    char_end: int
    text: str

    @classmethod
    def from_result(cls, result: SearchResult) -> ChunkOut:
        return cls(rank=result.rank, score=round(result.score, 4), **result.chunk.to_dict())


class SearchResponse(BaseModel):
    query: str
    results: list[ChunkOut]


class AskResponse(BaseModel):
    question: str
    mode: Literal["rag", "retrieval_only"]
    answer: str | None
    answer_found: bool
    citations: list[ChunkOut]
    results: list[ChunkOut]
    notice: str | None = Field(
        default=None, description="Why there is no generated answer, as a stable code"
    )
    notice_message: str | None = Field(default=None, description="The notice, localized")


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    chunks: int
    duplicate: bool
    skipped_pages: list[int] = Field(
        default_factory=list, description="1-based PDF pages that could not be read"
    )


class DocumentOut(BaseModel):
    doc_id: str
    filename: str
    chunks: int


class HealthResponse(BaseModel):
    status: Literal["ok"]


class StatsResponse(BaseModel):
    documents: int
    max_documents: int
    chunks: int
    index_backend: str
    llm_enabled: bool
    llm_provider: str | None = None
    llm_model: str | None = None


class ErrorResponse(BaseModel):
    error: str = Field(description="Stable error code")
    message: str = Field(description="Localized, user-facing message")
    fields: list[str] | None = Field(default=None, description="Invalid fields (422 only)")
