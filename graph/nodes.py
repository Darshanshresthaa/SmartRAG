"""
Node functions for the RAG LangGraph pipeline.

Each function takes the shared `State` and returns a partial state update,
following the standard LangGraph node signature.
"""

from langchain_core.documents import Document
from langgraph.types import interrupt

from Schema_Class.class_schema import State, Decision, QueryRewrite
from prompts.decision_prompt import decision_prompt
from prompts.llm_generate_prompt import llm_generate_prompt
from prompts.query_review_prompt import query_rewrite_prompt
from prompts.answer_prompt import answer_prompt
from service_llm.model import get_llm, get_tavily_search
from vector_db.vector_store import get_retriever

model = get_llm()
tavily_client = get_tavily_search()


def _format_docs(docs: list[Document]) -> str:
    """Render retrieved documents into the TITLE/AUTHOR/SOURCE/CONTENT block
    expected by the prompts."""
    return "\n\n".join(
        f"""
    TITLE: {doc.metadata.get('title')}
    AUTHOR: {doc.metadata.get('author')}
    SOURCE: {doc.metadata.get('source')}
    CONTENT: {doc.page_content}
    """
        for doc in docs
    )


def retrieve_node(state: State):
    """Retrieve the top-k relevant documents from the vector database."""
    retriever = get_retriever(k=5)
    docs = retriever.invoke(state.question)
    return {"retrievers": docs}


def decider_node(state: State):
    """Decide whether the retrieved context is sufficient to answer,
    or whether a web search is needed."""
    content = _format_docs(state.retrievers)

    decider_chain = decision_prompt | model.with_structured_output(Decision)
    result = decider_chain.invoke({"question": state.question, "context": content})

    return {"decision": result.decision}


def router_node(state: State):
    """Route to the document-answer or web-search branch based on decision."""
    if state.decision == "answer":
        return "doc_answer"
    elif state.decision == "web_search":
        return "web_search"


def generate_llm_node(state: State):
    """Generate the final answer using only the retrieved documents."""
    llm_chain = llm_generate_prompt | model
    content = _format_docs(state.retrievers)

    result = llm_chain.invoke({"question": state.question, "context": content})
    return {"final_answer": result.content}


def web_search_decider_node(state: State):
    """Human-in-the-loop checkpoint: ask the user for approval before
    falling back to a web search."""
    approval = interrupt(
        {
            "type": "approval",
            "title": "Web_search",
            "question": state.question,
            "message": (
                "The uploaded document isn't enough to answer your question.\n\n"
                "Would you like me to do a web search? (yes/no)"
            ),
        }
    )

    approval = str(approval).lower().strip()

    if approval == "yes":
        return {"web_search": "yes"}

    return {"web_search": "no"}


def hitl_router(state: State):
    """Route based on the user's web-search approval decision."""
    if state.web_search == "yes":
        return "web_search"
    return "display"


def enhance_query_node(state: State):
    """Rewrite the user's query to be better optimized for web search."""
    enhance_query_chain = query_rewrite_prompt | model.with_structured_output(QueryRewrite)
    result = enhance_query_chain.invoke({"query": state.question})

    state.web.enhanced_query = result.enhanced_query
    return {"web": state.web}


def web_search_node(state: State):
    """Run the web search using the enhanced query via Tavily."""
    response = tavily_client.invoke({"query": state.web.enhanced_query})

    docs = [
        Document(
            page_content=result.get("content", ""),
            metadata={
                "title": result.get("title", ""),
                "url": result.get("url", ""),
            },
        )
        for result in response
    ]

    state.web.raw_content = docs
    return {"web": state.web}


def summarize_web_search(state: State):
    """Summarize each web search result into a clean, structured form."""
    chain = answer_prompt | model

    summaries = []
    for doc in state.web.raw_content:
        result = chain.invoke(
            {
                "title": doc.metadata.get("title", ""),
                "url": doc.metadata.get("url", ""),
                "content": doc.page_content,
            }
        )
        summaries.append(result.content)

    state.web.summarize_answer = "\n\n".join(summaries)
    return {"web": state.web}


def display(state: State):
    """Produce the final result shown to the user."""
    if state.decision == "answer":
        return {"result": state.final_answer}

    elif state.decision == "web_search":
        if state.web_search == "no":
            return {
                "result": (
                    "Search cancelled — no web search was performed, "
                    "and the document didn't contain enough information to answer."
                )
            }
        elif state.web_search == "yes":
            return {"result": state.web.summarize_answer}
