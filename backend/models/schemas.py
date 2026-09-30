from pydantic import BaseModel, Field


class DocumentResponse(BaseModel):
    document_id: str
    filename: str


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


class SourceResponse(BaseModel):
    filename: str
    page: int
    distance: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
