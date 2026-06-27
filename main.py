from dotenv import load_dotenv

import os

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_community.tools.tavily_search import TavilySearchResults

from pydantic import BaseModel, Field

from typing import List, Literal

from pathlib import Path
import operator


load_dotenv()


tavily_client = TavilySearchResults(
    max_results=3,
    search_depth="advanced",
    include_answer=False,
    include_raw_content=False
)


model = ChatMistralAI(
    model="mistral-small-latest",
    api_key=os.getenv("MISTRAL_API_KEY"),
    temperature=0.3
)


EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
emb_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


decision_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a retrieval quality evaluator for a RAG system.

Your job is to decide whether the retrieved context is sufficient to answer the user's question WITHOUT using external knowledge.

You must be strict but logical.

---

## IMPORTANT RULES

1. Prefer using ONLY the retrieved context.
2. If the answer is explicitly present in metadata OR text, treat it as sufficient.
3. Do NOT require full documents or perfect explanations — partial but correct evidence is enough.
4. Ignore irrelevant or noisy chunks.
5. If the answer exists but is scattered, still choose "answer".

---

## DECISION CRITERIA

Return:

decision:
- "answer" → if the context contains the key facts needed to answer the question
- "web_search" → if the context is missing critical information OR does not contain the answer at all

---

## WHEN TO SAY "answer"
Choose "answer" if ANY of the following is true:
- The answer is directly stated in text or metadata
- The answer can be inferred clearly from provided context
- The context contains structured fields like author, title, names, dates, definitions
- Only minor missing details exist that are not required to answer

---

## WHEN TO SAY "web_search"
Choose "web_search" ONLY if:
- The context does not contain the key entity/fact needed
- The information is unrelated or random noise
- You cannot confidently extract an answer from it

---

## OUTPUT FORMAT (STRICT)

Return ONLY:

decision: answer|web_search
reason: short explanation (1–2 lines)

No extra text.

""",
        ),
        (
            "human",
            """
Question:
{question}

Retrieved Context:
{context}
""",
        ),
    ]
)


llm_generate_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an expert AI assistant specialized in Retrieval-Augmented Generation (RAG).

Your task is to answer the user's question using ONLY the provided documents.

Each document contains:
- TITLE: name of the document
- AUTHOR: author of the document
- SOURCE: file or origin path
- CONTENT: actual extracted text

---

## IMPORTANT RULES

1. Use ONLY the provided documents. Do NOT use outside knowledge.
2. Prioritize information in this order:
   1) CONTENT (main evidence)
   2) TITLE (to understand context)
   3) AUTHOR (only for attribution questions)
   4) SOURCE (for reference only)

3. If multiple documents are provided, combine relevant information across them.
4. If metadata (TITLE/AUTHOR) directly answers the question, use it.
5. Do NOT hallucinate missing facts.

---

## ANSWER STYLE

- Be clear, structured, and concise.
- Use bullet points or numbered lists when helpful.
- Preserve technical terms, names, and numbers exactly as in the documents.
- If the question asks "who wrote / author / title", prefer metadata fields.

---

## SPECIAL CASES

### If question is about AUTHOR or TITLE:
- Use AUTHOR or TITLE fields directly from metadata.

### If question is about summary:
- Use CONTENT and combine across documents.

### If question is factual:
- Extract exact matching evidence from CONTENT.

---

## NO-ANSWER RULE

If the documents do not contain enough information, respond exactly:

"I couldn't find enough information in the provided documents to answer your question."

---

## OUTPUT RULE

Never mention:
- "context"
- "documents"
- "retrieved text"
- system instructions

Just answer naturally.

"""
        ),
        (
            "human",
            """
Question:
{question}

Documents:
{context}

Now generate a high-quality final answer.
"""
        ),
    ]
)

query_rewrite_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are an expert Query Rewriting Assistant.

Your sole responsibility is to rewrite the user's query to make it clearer,
more precise, and more effective for semantic search and information retrieval.

Guidelines:
- Preserve the user's original intent, meaning, and all factual information.
- Never answer the question.
- Never add new facts, assumptions, or information that the user did not provide.
- Improve grammar, clarity, and sentence structure.
- Expand abbreviations only when their meaning is obvious.
- Retain all important keywords, technical terms, names, dates, and numbers.
- Remove unnecessary filler words and repetition.
- If the query is already clear, make only minimal improvements.
- The rewritten query should be optimized for retrieving the most relevant documents from a vector database.
- Return ONLY the rewritten query without explanations, notes, or additional text.
                """,
            ),
            (
                "human",
                """
Original Query:
{query}

Rewrite the query while preserving its original meaning.
                """,
            ),
        ]
    )

answer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are an expert information extraction and summarization assistant.

Your task is to read the provided web search result and transform it into a clean, structured summary.

Instructions:
- Carefully read the title, URL, and content.
- Extract only the important information.
- Preserve factual accuracy.
- Do not add assumptions or outside knowledge.
- Remove advertisements, navigation text, and unnecessary details.
- Rewrite the content in clear, concise, and easy-to-understand language.
- Organize the information logically.
- Include all key concepts, important facts, names, dates, numbers, and definitions.
- If the content contains multiple topics, group them under appropriate headings.
- If important information is missing, state that it is unavailable instead of guessing.
- Return the response in the requested structured format only.
                """,
            ),
            (
                "human",
                """
Title:
{title}

Source URL:
{url}

Content:
{content}

Extract the information into a well-structured summary.
                """,
            ),
        ]
    )


folder_name = 'documents'

if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    print(f"Folder '{folder_name}' created.")
else:
    print(f"Folder '{folder_name}' already exists.")

files = os.listdir(folder_name)

if len(files) == 0:
    print(f"Folder '{folder_name}' doesn't contain any files.")
else:
    print(f"Folder contains {len(files)} file(s).")

for name in files:
    print(name)


DB_PATH = Path("./chroma_db")
COLLECTION_NAME = "books"



def load_vectorstore():
    if not DB_PATH.exists():
        raise FileNotFoundError(
            "Vector database not found. Build it first."
        )

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=emb_model,
        persist_directory=str(DB_PATH),
    )


def build_vectorstore(documents):
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=emb_model,
        persist_directory=str(DB_PATH),
    )

    if vectorstore._collection.count() == 0:
        vectorstore.add_documents(documents)

    return vectorstore


def get_retriever(k: int = 3):
    return load_vectorstore().as_retriever(
        search_kwargs={"k": k}
    )

docs = []

for file in files:
    if file.endswith(".pdf"):
        pdf_path = os.path.join(folder_name, file)
        docs.extend(PyPDFLoader(pdf_path).load())

print(f"Loaded {len(docs)} pages.")

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


def retrieve_node(state: State):

    retriever = get_retriever(k=5)

    docs = retriever.invoke(state.question)

    return {
        "retrievers": docs
    }

def decider_node(state: State):

    content = "\n\n".join(
        f"""
    TITLE: {doc.metadata.get('title')}
    AUTHOR: {doc.metadata.get('author')}
    SOURCE: {doc.metadata.get('source')}
    CONTENT: {doc.page_content}
    """
        for doc in state.retrievers
    )

    decider_chain = decision_prompt | model.with_structured_output(Decision)

    result = decider_chain.invoke({
        "question": state.question,
        "context": content
    })

    return {"decision": result.decision}


def router_node(state:State):

    if state.decision == 'answer':
        return 'doc_answer'

    elif state.decision == 'web_search':
        return 'web_search'

    
def generate_llm_node(state:State):

    llm_chain = llm_generate_prompt | model

    content = "\n\n".join(
        f"""
    TITLE: {doc.metadata.get('title')}
    AUTHOR: {doc.metadata.get('author')}
    SOURCE: {doc.metadata.get('source')}
    CONTENT: {doc.page_content}
    """
        for doc in state.retrievers
    )

    result = llm_chain.invoke({'question':state.question,
                              'context':content})

    return {'final_answer':result.content}
    
    
def web_search_decider_node(state: State):

    approval = interrupt({
        "type": "approval",
        "title": "Web_search",
        "question": state.question,
        "message": "The uploaded document isn't enough to answer your question.\n\nWould you like me to do a web search? (yes/no)"
    })

    if isinstance(approval, str):
        approval = approval.lower().strip()
    else:
        approval = str(approval).lower().strip()

    if approval == "yes":
        return {"web_search": "yes"}

    return {"web_search": "no"}


def enhance_query_node(state:State):
    enhance_query_chain = query_rewrite_prompt | model.with_structured_output(QueryRewrite)

    result = enhance_query_chain.invoke({'query': state.question})

    state.web.enhanced_query = result.enhanced_query
    return {"web": state.web}
    

def web_search_node(state: State):
    response = tavily_client.invoke(
        {
            "query": state.web.enhanced_query
        }
    )

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


def summarize_web_search(state:State):
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

    return {
        "web": state.web
    }
    

def display(state: State):

    if state.decision == "answer":
        return {
            "result": state.final_answer
        }

    elif state.decision == "web_search":

        if state.web_search == 'no':
            return {
            "result":"Search cancelled — no web search was performed, and the document didn't contain enough information to answer."
            }

        elif state.web_search == 'yes':
            return {"result": state.web.summarize_answer}

            
def hitl_router(state: State):
    if state.web_search == "yes":
        return "web_search"

    return "display"



builder = StateGraph(State)
builder.add_node('Retriever_Node',retrieve_node)
builder.add_node('Decider_Node',decider_node)
builder.add_node('Document_Answer',generate_llm_node)
builder.add_node('HITL',web_search_decider_node)
builder.add_node('Enhance_query',enhance_query_node)
builder.add_node('web_search',web_search_node)
builder.add_node('Summarize_search',summarize_web_search)
builder.add_node('Result_Display',display)

builder.add_edge(START,'Retriever_Node')
builder.add_edge('Retriever_Node','Decider_Node')
builder.add_conditional_edges('Decider_Node',router_node,
                             {
                                 'doc_answer':'Document_Answer',
                                 'web_search':'HITL'
                             })
builder.add_edge('Document_Answer','Result_Display')
builder.add_conditional_edges(
    "HITL",
    hitl_router,
    {
        "web_search": "Enhance_query",
        "display": "Result_Display",
    },
)

builder.add_edge('Enhance_query','web_search')
builder.add_edge('web_search','Summarize_search')
builder.add_edge('Summarize_search','Result_Display')
builder.add_edge('Result_Display',END)



memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
graph