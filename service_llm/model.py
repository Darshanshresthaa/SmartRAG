from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_mistralai import ChatMistralAI
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import os

load_dotenv()


def get_tavily_search():
    return TavilySearchResults(
        max_results=3,
        search_depth="advanced",
        include_answer=False,
        include_raw_content=False,
    )


def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.3,
    )


def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )