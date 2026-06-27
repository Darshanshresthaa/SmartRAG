"""
Builds and compiles the RAG StateGraph.

Wires together the node functions defined in graph.nodes into the
full retrieve -> decide -> (answer | HITL web search) -> display pipeline.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from Schema_Class.class_schema import State
from graph.nodes import (
    retrieve_node,
    decider_node,
    router_node,
    generate_llm_node,
    web_search_decider_node,
    hitl_router,
    enhance_query_node,
    web_search_node,
    summarize_web_search,
    display,
)


def build_graph():
    """Construct and compile the RAG pipeline graph with an in-memory checkpointer."""
    builder = StateGraph(State)

    builder.add_node("Retriever_Node", retrieve_node)
    builder.add_node("Decider_Node", decider_node)
    builder.add_node("Document_Answer", generate_llm_node)
    builder.add_node("HITL", web_search_decider_node)
    builder.add_node("Enhance_query", enhance_query_node)
    builder.add_node("web_search", web_search_node)
    builder.add_node("Summarize_search", summarize_web_search)
    builder.add_node("Result_Display", display)

    builder.add_edge(START, "Retriever_Node")
    builder.add_edge("Retriever_Node", "Decider_Node")

    builder.add_conditional_edges(
        "Decider_Node",
        router_node,
        {
            "doc_answer": "Document_Answer",
            "web_search": "HITL",
        },
    )

    builder.add_edge("Document_Answer", "Result_Display")

    builder.add_conditional_edges(
        "HITL",
        hitl_router,
        {
            "web_search": "Enhance_query",
            "display": "Result_Display",
        },
    )

    builder.add_edge("Enhance_query", "web_search")
    builder.add_edge("web_search", "Summarize_search")
    builder.add_edge("Summarize_search", "Result_Display")
    builder.add_edge("Result_Display", END)

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


graph = build_graph()
