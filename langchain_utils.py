"""
RAG pipeline built with LangGraph instead of the deprecated
`langchain.chains.create_history_aware_retriever` / `create_retrieval_chain`.

Graph shape:

    START -> contextualize -> retrieve -> generate -> END

- contextualize: rewrites the latest user question into a standalone question
  using chat history (skipped when history is empty).
- retrieve: runs the standalone question through the Chroma retriever.
- generate: stuffs retrieved docs into a QA prompt and calls the LLM.

The graph exposes the same input/output shape the old code did, so main.py
can still call:  rag_chain.invoke({"input": ..., "chat_history": ...})["answer"]
"""

from typing import List, TypedDict
from functools import lru_cache
import os

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langgraph.graph import StateGraph, START, END

from chroma_utils import vectorstore

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("GROQ_API_KEY environment variable is not set.")


# ---------- Graph state ----------
class RagState(TypedDict, total=False):
    input: str                    # raw user question
    chat_history: List[dict]      # [{"role": "human"|"ai", "content": "..."}]
    standalone_question: str      # question after contextualization
    context: List[Document]       # retrieved docs
    answer: str                   # final answer


# ---------- Prompts ----------
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI assistant. Use the following context to answer the user's question."),
        ("system", "Context: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)


# ---------- Helpers ----------
def _to_lc_messages(history: List[dict]) -> List[BaseMessage]:
    """Convert db_utils history format to LangChain message objects."""
    messages: List[BaseMessage] = []
    for m in history or []:
        role = m.get("role")
        content = m.get("content", "")
        if role == "human":
            messages.append(HumanMessage(content=content))
        elif role == "ai":
            messages.append(AIMessage(content=content))
    return messages


def _format_docs(docs: List[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)


# ---------- Graph factory ----------
@lru_cache(maxsize=64)
def get_rag_chain(model: str, user_id: int):
    """
    Build and compile the LangGraph RAG pipeline for (model, user_id).
    The retriever is scoped to only this user's document chunks.
    Cached so we don't rebuild the graph on every request.
    """
    llm = ChatGroq(model=model, temperature=0)

    # Retriever with per-user metadata filter — this is what actually
    # enforces document isolation at query time.
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 2, "filter": {"user_id": user_id}}
    )

    contextualize_chain = contextualize_q_prompt | llm | StrOutputParser()
    answer_chain = qa_prompt | llm | StrOutputParser()

    # ---- Nodes ----
    def contextualize_node(state: RagState) -> RagState:
        history = _to_lc_messages(state.get("chat_history", []))
        question = state["input"]
        # If there's no history, there's nothing to contextualize.
        if not history:
            return {"standalone_question": question}

        rewritten = contextualize_chain.invoke(
            {"input": question, "chat_history": history}
        )
        return {"standalone_question": rewritten}

    def retrieve_node(state: RagState) -> RagState:
        query = state.get("standalone_question") or state["input"]
        docs = retriever.invoke(query)
        return {"context": docs}

    def generate_node(state: RagState) -> RagState:
        history = _to_lc_messages(state.get("chat_history", []))
        answer = answer_chain.invoke(
            {
                "input": state["input"],
                "chat_history": history,
                "context": _format_docs(state.get("context", [])),
            }
        )
        return {"answer": answer}

    # ---- Wire up the graph ----
    graph = StateGraph(RagState)
    graph.add_node("contextualize", contextualize_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)

    graph.add_edge(START, "contextualize")
    graph.add_edge("contextualize", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()