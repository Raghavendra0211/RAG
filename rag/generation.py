"""
Generation Module for Learn RAG.
Connects to OpenRouter API (configurable, defaults to recursive/flint-v1.0:free),
enforces strict context grounding, handles model fallbacks,
and bundles answers with source citations.
"""

from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from rag.config import mask_secret
from rag.retrieval import format_citations


GROUNDED_SYSTEM_PROMPT = """You are an accurate, strictly grounded AI teaching assistant for a technical RAG workshop.
Your mission is to answer questions using EXCLUSIVELY the provided context passages from the uploaded document.

CRITICAL RULES:
1. Answer the user's question using the provided document context passages. Extract, summarize, and synthesize any relevant details, explanations, features, or uses explicitly present in the text.
2. Do NOT use outside knowledge, prior assumptions, or extrapolate beyond what is explicitly stated in the context.
3. If the provided context contains NO relevant information or mentions nothing about the subject of the question, reply with:
   "The provided document does not contain information to answer this question."
4. Keep the answer direct, factual, and well-explained for students. Do not invent or hallucinate facts or citations."""

STANDARD_REFUSAL_MESSAGE = "The provided document does not contain information to answer this question."


def build_context_string(chunks: List[Document]) -> str:
    """
    Formats retrieved chunks into a clearly labeled context block with page indicators.
    """
    if not chunks:
        return ""

    formatted_parts: List[str] = []
    for idx, chunk in enumerate(chunks, 1):
        filename = chunk.metadata.get(
            "filename", chunk.metadata.get("source", "Document.pdf")
        )
        page = chunk.metadata.get("page", chunk.metadata.get("page_number", 1))
        content = chunk.page_content.strip()

        part = f"--- [Passage {idx} | Source: {filename} | Page {page}] ---\n{content}"
        formatted_parts.append(part)

    return "\n\n".join(formatted_parts)


def get_llm_client(
    openrouter_api_key: str,
    model_name: str = "recursive/flint-v1.0:free",
    temperature: float = 0.2,
) -> ChatOpenAI:
    """
    Creates and returns a LangChain ChatOpenAI instance configured for OpenRouter.
    """
    return ChatOpenAI(
        model=model_name,
        openai_api_key=openrouter_api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=temperature,
        max_tokens=1024,
        default_headers={
            "HTTP-Referer": "https://github.com/learn-rag-workshop",
            "X-Title": "Learn RAG Workshop",
        },
    )


def generate_grounded_answer(
    query: str,
    retrieved_chunks: List[Document],
    openrouter_api_key: str,
    model_name: str = "recursive/flint-v1.0:free",
    fallback_model: str = "meta-llama/llama-3.1-8b-instruct",
) -> Dict[str, Any]:
    """
    Sends retrieved document context and question to the OpenRouter chat model.
    Enforces strict context grounding and handles model fallback gracefully.
    """
    citations = format_citations(retrieved_chunks)

    # Empty retrieval check: immediately refuse without spending API tokens
    if not retrieved_chunks:
        return {
            "answer": STANDARD_REFUSAL_MESSAGE,
            "citations": citations,
            "chunks": [],
            "model_used": model_name,
            "grounded": False,
        }

    context_text = build_context_string(retrieved_chunks)

    user_prompt = f"""DOCUMENT CONTEXT:
{context_text}

USER QUESTION:
{query}

ANSWER (grounded strictly in the document context above):"""

    messages = [
        SystemMessage(content=GROUNDED_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    selected_model = model_name
    try:
        llm = get_llm_client(openrouter_api_key=openrouter_api_key, model_name=selected_model)
        response = llm.invoke(messages)
    except Exception as exc:
        err_msg = str(exc)
        # Check if error is due to unavailable model or 404
        is_model_unavailable = (
            "404" in err_msg
            or "not found" in err_msg.lower()
            or "not available" in err_msg.lower()
            or "unavailable" in err_msg.lower()
        )
        if is_model_unavailable and fallback_model and selected_model != fallback_model:
            selected_model = fallback_model
            fallback_llm = get_llm_client(
                openrouter_api_key=openrouter_api_key,
                model_name=selected_model,
                temperature=0.2,
            )
            response = fallback_llm.invoke(messages)
        else:
            # Mask API key if it appears in any error message to protect secrets
            safe_err = err_msg.replace(openrouter_api_key, mask_secret(openrouter_api_key))
            raise RuntimeError(f"OpenRouter generation error: {safe_err}") from exc

    raw_answer = response.content
    if isinstance(raw_answer, list):
        # In case content is returned as structured parts
        answer_text = "".join(str(p) for p in raw_answer).strip()
    else:
        answer_text = str(raw_answer).strip()

    # Check for refusal phrases and normalize to exact required standard string
    lower_answer = answer_text.lower()
    is_refusal = (
        "does not contain information" in lower_answer
        or "not found in the provided" in lower_answer
        or "context does not mention" in lower_answer
        or "not mentioned in the provided" in lower_answer
        or "provided document does not contain" in lower_answer
    )

    if is_refusal:
        final_answer = STANDARD_REFUSAL_MESSAGE
        grounded = False
    else:
        final_answer = answer_text
        grounded = True

    return {
        "answer": final_answer,
        "citations": citations,
        "chunks": retrieved_chunks,
        "model_used": selected_model,
        "grounded": grounded,
    }
