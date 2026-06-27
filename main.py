"""
Entrypoint for the RAG pipeline.

Loads any PDFs in documents/, builds (or reuses) the vectorstore, then
runs a question through the compiled LangGraph pipeline.
"""

from dotenv import load_dotenv

from ingestion.loader import ensure_documents_folder, load_pdfs
from vector_db.vector_store import build_vectorstore
from graph.builder import build_graph
from langgraph.types import Command

load_dotenv()


def setup():
    """Ensure documents are loaded and the vectorstore is built."""
    ensure_documents_folder()
    docs = load_pdfs()
    build_vectorstore(docs)


def ask(question: str, thread_id: str = "default-thread"):
    graph = build_graph()
    config = {"configurable": {"thread_id": thread_id}}

    result = graph.invoke(
        {"question": question},
        config=config,
    )

    while "__interrupt__" in result:
        user_input = input(
            "The document doesn't contain enough information.\n"
            "Would you like to perform a web search? (yes/no): "
        ).strip().lower()

        result = graph.invoke(
            Command(resume=user_input),
            config=config,
        )

    return result["result"]





def main():
    setup()
    question = input("Ask a question: ")
    answer = ask(question)
    print("\n--- ANSWER ---\n")
    print(answer)


if __name__ == "__main__":
    main()
