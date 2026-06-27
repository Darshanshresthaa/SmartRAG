"""
Entrypoint for the RAG pipeline.

Loads any PDFs in documents/, builds (or reuses) the vectorstore, then
runs a question through the compiled LangGraph pipeline.
"""

from dotenv import load_dotenv

from ingestion.loader import ensure_documents_folder, load_pdfs
from vector_db.vector_store import build_vectorstore
from graph.builder import build_graph

load_dotenv()


def setup():
    """Ensure documents are loaded and the vectorstore is built."""
    ensure_documents_folder()
    docs = load_pdfs()
    build_vectorstore(docs)


def ask(question: str, thread_id: str = "default-thread"):
    """Run a question through the RAG graph and return the result."""
    graph = build_graph()
    config = {"configurable": {"thread_id": thread_id}}

    result = graph.invoke({"question": question}, config=config)

    # If the graph paused for human-in-the-loop web search approval,
    # surface that to the caller instead of silently returning nothing.
    if "__interrupt__" in result:
        return result

    return result.get("result")


def main():
    setup()
    question = input("Ask a question: ")
    answer = ask(question)
    print("\n--- ANSWER ---\n")
    print(answer)


if __name__ == "__main__":
    main()
