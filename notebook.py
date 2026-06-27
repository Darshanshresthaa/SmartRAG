# RagBasic — interactive notebook entrypoint
#
# This file is kept thin on purpose: all real logic now lives in the
# package modules (Schema_Class/, prompts/, service_llm/, vector_db/,
# ingestion/, graph/). Run cells here to explore the pipeline interactively;
# edit the actual logic in those modules, not here.

from dotenv import load_dotenv

load_dotenv()

# %%
from ingestion.loader import ensure_documents_folder, load_pdfs
from vector_db.vector_store import build_vectorstore, get_retriever
from graph.builder import build_graph

# %% Load documents and build the vectorstore
ensure_documents_folder()
docs = load_pdfs()
vectorstore = build_vectorstore(docs)

# %% Compile the graph
graph = build_graph()
graph

# %% Try a question
config = {"configurable": {"thread_id": "notebook-thread"}}
result = graph.invoke({"question": "Replace with your question"}, config=config)
result
