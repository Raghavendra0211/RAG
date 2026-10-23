"""
Unit & Pipeline Tests for Learn RAG Application.
Validates chunking, metadata preservation, 1-indexed page numbers,
citation formatting, configuration checks, document isolation,
grounding prompts, and refusal behavior without live external APIs.
"""

from unittest.mock import MagicMock, patch
import pytest
from langchain_core.documents import Document

from rag.config import AppConfig, validate_config, mask_secret
from rag.ingestion import (
    compute_document_id,
    split_document_pages,
    load_pdf_from_bytes,
)
from rag.retrieval import (
    format_citations,
    retrieve_relevant_chunks,
)
from rag.generation import (
    build_context_string,
    generate_grounded_answer,
    GROUNDED_SYSTEM_PROMPT,
    STANDARD_REFUSAL_MESSAGE,
)


def test_config_validation_missing_keys():
    """Test that missing keys or invalid configurations are flagged."""
    cfg_empty = AppConfig(
        qdrant_url="",
        qdrant_api_key="",
        openrouter_api_key="",
    )
    is_valid, issues = validate_config(cfg_empty)
    assert not is_valid
    assert any("QDRANT_URL" in issue for issue in issues)
    assert any("QDRANT_API_KEY" in issue for issue in issues)
    assert any("OPENROUTER_API_KEY" in issue for issue in issues)


def test_config_validation_localhost_rejected():
    """Test that localhost/127.0.0.1 Qdrant URLs are rejected per cloud requirements."""
    cfg_localhost = AppConfig(
        qdrant_url="http://localhost:6333",
        qdrant_api_key="valid_key",
        openrouter_api_key="valid_key",
    )
    is_valid, issues = validate_config(cfg_localhost)
    assert not is_valid
    assert any("localhost" in issue for issue in issues)


def test_config_validation_placeholder_rejected():
    """Test that default template placeholder values from .env.example are rejected."""
    cfg_placeholder = AppConfig(
        qdrant_url="https://your-cluster-id.us-east-1-0.aws.cloud.qdrant.io:6333",
        qdrant_api_key="your_qdrant_api_key_here",
        openrouter_api_key="your_openrouter_api_key_here",
    )
    is_valid, issues = validate_config(cfg_placeholder)
    assert not is_valid
    assert len(issues) >= 3


def test_config_validation_valid_cloud():
    """Test that a well-formed cloud configuration passes validation."""
    cfg_valid = AppConfig(
        qdrant_url="https://c8310b8-abc.us-east4.gcp.cloud.qdrant.io:6333",
        qdrant_api_key="sec_abcdef1234567890",
        openrouter_api_key="sk-or-v1-abcdef1234567890",
        openrouter_model="recursive/flint-v1.0:free",
    )
    is_valid, issues = validate_config(cfg_valid)
    assert is_valid
    assert len(issues) == 0


def test_mask_secret():
    """Test that secrets are safely masked to prevent accidental credential leakage."""
    assert mask_secret("") == "<not set>"
    assert mask_secret("short") == "***"
    masked = mask_secret("sk-or-v1-1234567890abcdef")
    assert masked.startswith("sk-o")
    assert masked.endswith("cdef")
    assert "..." in masked


def test_compute_document_id():
    """Test deterministic document hash generation from binary contents."""
    sample_bytes_1 = b"%PDF-1.4 Mock PDF Content A"
    sample_bytes_2 = b"%PDF-1.4 Mock PDF Content B"

    hash_1a = compute_document_id(sample_bytes_1, "file1.pdf")
    hash_1b = compute_document_id(sample_bytes_1, "file1.pdf")
    hash_2 = compute_document_id(sample_bytes_2, "file2.pdf")

    assert hash_1a == hash_1b
    assert hash_1a != hash_2
    assert len(hash_1a) == 16

    with pytest.raises(ValueError, match="empty bytes"):
        compute_document_id(b"")


def test_empty_pdf_handling():
    """Test that 0-byte or corrupted inputs raise descriptive ValueErrors."""
    with pytest.raises(ValueError, match="empty"):
        load_pdf_from_bytes(b"", "empty.pdf")


def test_split_document_pages_chunking():
    """Test that text splitting adheres to chunk size 1000 and overlap 400."""
    page_1_text = "Word " * 300  # ~1500 chars
    page_2_text = "Section " * 250  # ~2000 chars

    docs = [
        Document(page_content=page_1_text, metadata={"page": 1, "filename": "test.pdf"}),
        Document(page_content=page_2_text, metadata={"page": 2, "filename": "test.pdf"}),
    ]

    doc_id = "test_doc_123"
    chunks = split_document_pages(
        documents=docs,
        doc_id=doc_id,
        filename="test.pdf",
        chunk_size=1000,
        chunk_overlap=400,
    )

    assert len(chunks) > 2
    for chunk in chunks:
        assert len(chunk.page_content) <= 1000
        assert chunk.metadata["doc_id"] == doc_id
        assert chunk.metadata["filename"] == "test.pdf"
        assert chunk.metadata["page"] in [1, 2]
        assert "chunk_id" in chunk.metadata
        assert chunk.metadata["chunk_id"].startswith(f"{doc_id}_chunk_")


def test_citation_formatting():
    """Test that citations are deduplicated and formatted with human-friendly page numbers."""
    chunks = [
        Document(
            page_content="Content from page 3",
            metadata={"filename": "lecture_notes.pdf", "page": 3, "similarity_score": 0.88},
        ),
        Document(
            page_content="Another piece from page 1",
            metadata={"filename": "lecture_notes.pdf", "page": 1, "similarity_score": 0.92},
        ),
        Document(
            page_content="Second piece from page 3",
            metadata={"filename": "lecture_notes.pdf", "page": 3, "similarity_score": 0.85},
        ),
    ]

    citations = format_citations(chunks)

    assert "lecture_notes.pdf" in citations["citations_by_file"]
    assert citations["citations_by_file"]["lecture_notes.pdf"] == [1, 3]
    assert "lecture_notes.pdf (Pages 1, 3)" in citations["citation_summary"]
    assert len(citations["source_details"]) == 3


def test_citation_formatting_empty():
    """Test citation formatting when no chunks are retrieved."""
    citations = format_citations([])
    assert "No citations" in citations["citation_summary"]
    assert citations["source_details"] == []


def test_context_builder_labels():
    """Test context block construction contains source filename and page numbers."""
    chunks = [
        Document(
            page_content="Retrieval Augmented Generation bridges LLMs with external data.",
            metadata={"filename": "rag_intro.pdf", "page": 2},
        ),
        Document(
            page_content="Embeddings represent semantic meaning in dense vector space.",
            metadata={"filename": "rag_intro.pdf", "page": 4},
        ),
    ]

    context_str = build_context_string(chunks)
    assert "Passage 1 | Source: rag_intro.pdf | Page 2" in context_str
    assert "Passage 2 | Source: rag_intro.pdf | Page 4" in context_str
    assert "Retrieval Augmented Generation bridges" in context_str


def test_empty_chunks_grounded_fallback():
    """Test that asking with no retrieved chunks immediately yields the grounded fallback without calling LLM."""
    result = generate_grounded_answer(
        query="What is the speed of light?",
        retrieved_chunks=[],
        openrouter_api_key="mock_key",
        model_name="recursive/flint-v1.0:free",
    )

    assert result["answer"] == STANDARD_REFUSAL_MESSAGE
    assert result["grounded"] is False
    assert result["chunks"] == []


def test_grounded_system_prompt_rules():
    """Test that the system prompt strictly forbids hallucination and enforces document-only answers."""
    assert "EXCLUSIVELY" in GROUNDED_SYSTEM_PROMPT
    assert STANDARD_REFUSAL_MESSAGE in GROUNDED_SYSTEM_PROMPT
    assert "Do not invent or hallucinate" in GROUNDED_SYSTEM_PROMPT


def test_grounded_answer_refusal_normalization():
    """Test that when the LLM responds with refusal language, it is normalized to standard refusal string."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Based on the context, the provided document does not contain information to answer this question."
    mock_llm.invoke.return_value = mock_response

    with patch("rag.generation.get_llm_client", return_value=mock_llm):
        chunks = [
            Document(page_content="Python is an interpreted programming language.", metadata={"page": 1})
        ]
        res = generate_grounded_answer(
            query="What is quantum gravity?",
            retrieved_chunks=chunks,
            openrouter_api_key="mock_key",
        )
        assert res["answer"] == STANDARD_REFUSAL_MESSAGE
        assert res["grounded"] is False


def test_document_scoped_retrieval_filtering():
    """Test that retrieval strictly enforces document-scoped filtering."""
    mock_client = MagicMock()
    mock_vector_store = MagicMock()

    doc_target = Document(
        page_content="Target document info",
        metadata={"doc_id": "target_doc_1", "page": 1, "filename": "target.pdf"},
    )
    doc_foreign = Document(
        page_content="Foreign document info",
        metadata={"doc_id": "other_doc_2", "page": 1, "filename": "other.pdf"},
    )

    mock_vector_store.similarity_search_with_score.return_value = [
        (doc_target, 0.88),
        (doc_foreign, 0.82),
    ]

    with patch("rag.retrieval.get_vector_store", return_value=mock_vector_store):
        results = retrieve_relevant_chunks(
            client=mock_client,
            embeddings_model=MagicMock(),
            collection_name="learning_rag",
            query="test query",
            doc_id="target_doc_1",
            top_k=4,
        )

        assert len(results) == 1
        assert results[0].metadata["doc_id"] == "target_doc_1"
        assert results[0].page_content == "Target document info"
        assert results[0].metadata["similarity_score"] == 0.88


def test_load_sample_pdf_and_page_indexing():
    """Test loading the included sample_nodejs.pdf and verifying 1-indexed page conversion."""
    import os
    sample_path = os.path.join(os.path.dirname(__file__), "..", "sample_nodejs.pdf")
    if os.path.exists(sample_path):
        with open(sample_path, "rb") as f:
            pdf_bytes = f.read()
        docs = load_pdf_from_bytes(pdf_bytes, "sample_nodejs.pdf")
        assert len(docs) == 2
        assert docs[0].metadata["page"] == 1
        assert docs[1].metadata["page"] == 2
        assert "Node.js" in docs[0].page_content
        assert "Module System" in docs[1].page_content
