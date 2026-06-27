from pydantic import BaseModel, Field
from langchain_core.documents import Document
from typing import List, Literal


class QueryRewrite(BaseModel):
    enhanced_query: str = Field(
        description="A rewritten version of the user's query optimized for retrieval."
    )

class WebSearch(BaseModel):
    enhanced_query: str | None = None

    raw_content: List[Document] = Field(
        default_factory=list,
        description="Documents returned from the web search."
    )

    summarize_answer: str | None = Field(
        default=None,
        description="Summary generated from the web search results."
    )

class Decision(BaseModel):
    decision: Literal["answer", "web_search"]
    reason: str

class State(BaseModel):

    question: str = Field(
        description="Original user question."
    )

    retrievers: List[Document] = Field(
        default_factory=list,
        description="Documents retrieved from the vector database."
    )

    decision: Literal["answer", "web_search"] | None = None

    web: WebSearch = Field(
        default_factory=WebSearch
    )

    web_search:Literal['yes','no'] | None = None

    final_answer: str | None = None

    result: str | None = Field(
        default=None,
        description="Final result to display to the user."
    )