"""
Live Cloud Infrastructure Validation Script for Learn RAG.
Verifies connectivity to Qdrant Cloud and OpenRouter API when real credentials are configured in .env.
Safe to run in CI/offline: exits gracefully if credentials are not configured.
"""

import os
import sys
import uuid

# Ensure project root is in sys.path when executed directly as a script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from rag.config import get_config, validate_config, mask_secret
from rag.ingestion import (
    get_qdrant_client,
    ensure_collection_exists,
    index_chunks_to_qdrant,
    delete_document_vectors,
)
from rag.retrieval import get_embedding_model, retrieve_relevant_chunks
from rag.generation import get_llm_client


def run_cloud_validation():
    print("=" * 65)
    print("   LEARN RAG — LIVE CLOUD INFRASTRUCTURE VALIDATION")
    print("=" * 65)

    config = get_config()
    is_valid, issues = validate_config(config)

    if not is_valid:
        print("\n[!] Cloud credentials are not fully configured yet.")
        print("    Configuration checks that need your attention in .env:")
        for issue in issues:
            print(f"    - {issue}")
        print("\n--> To run live cloud validation, populate your .env file with real credentials.")
        print("--> Unit tests (pytest tests/) can still be run locally without cloud credentials.")
        sys.exit(0)

    print("\n[+] Configuration validated successfully!")
    print(f"    Qdrant URL:      {config.qdrant_url}")
    print(f"    Qdrant Key:      {mask_secret(config.qdrant_api_key)}")
    print(f"    Collection:      {config.qdrant_collection_name}")
    print(f"    OpenRouter Key:  {mask_secret(config.openrouter_api_key)}")
    print(f"    OpenRouter Model:{config.openrouter_model}")

    # 1. Test Qdrant Cloud Connectivity & Operations
    print("\n[1/2] Testing Qdrant Cloud integration...")
    try:
        client = get_qdrant_client(config.qdrant_url, config.qdrant_api_key)
        ensure_collection_exists(client, config.qdrant_collection_name, vector_dim=384)
        print("      ✓ Connected to Qdrant Cloud cluster.")
        print(f"      ✓ Collection '{config.qdrant_collection_name}' verified.")

        # Test vector upload and cleanup
        test_doc_id = f"test_{uuid.uuid4().hex[:8]}"
        test_chunk = Document(
            page_content="Learn RAG cloud validation test string for vector similarity.",
            metadata={
                "doc_id": test_doc_id,
                "filename": "validation_test.pdf",
                "page": 1,
                "page_number": 1,
                "chunk_index": 0,
                "chunk_id": f"{test_doc_id}_chunk_0",
            },
        )

        embeddings = get_embedding_model(config.embedding_model_name)
        indexed_count = index_chunks_to_qdrant(
            client=client,
            embeddings_model=embeddings,
            collection_name=config.qdrant_collection_name,
            chunks=[test_chunk],
            doc_id=test_doc_id,
            overwrite_existing=True,
        )
        print(f"      ✓ Stored {indexed_count} test vector point(s).")

        # Test scoped similarity search
        results = retrieve_relevant_chunks(
            client=client,
            embeddings_model=embeddings,
            collection_name=config.qdrant_collection_name,
            query="Learn RAG validation test",
            doc_id=test_doc_id,
            top_k=1,
        )
        assert len(results) >= 1, "Expected at least 1 retrieved test chunk"
        print(f"      ✓ Scoped vector retrieval succeeded (score: {results[0].metadata.get('similarity_score', 'N/A'):.4f}).")

        # Cleanup test point
        deleted = delete_document_vectors(client, config.qdrant_collection_name, test_doc_id)
        print(f"      ✓ Cleaned up {deleted} temporary test vector(s).")
        print("      [PASS] Qdrant Cloud is fully operational!")

    except Exception as qdrant_err:
        print(f"      [FAIL] Qdrant Cloud test failed: {qdrant_err}")
        sys.exit(1)

    # 2. Test OpenRouter LLM Connectivity
    print("\n[2/2] Testing OpenRouter API integration...")
    try:
        llm = get_llm_client(
            openrouter_api_key=config.openrouter_api_key,
            model_name=config.openrouter_model,
        )
        response = llm.invoke([HumanMessage(content="Respond with 'READY' if you can read this message.")])
        reply = response.content.strip() if response.content else ""
        print(f"      ✓ Received response from model '{config.openrouter_model}': {reply[:60]}")
        print("      [PASS] OpenRouter API is fully operational!")
    except Exception as llm_err:
        print(f"      [!] Model '{config.openrouter_model}' returned an error: {llm_err}")
        print(f"      Testing fallback model '{config.openrouter_fallback_model}'...")
        try:
            fallback_llm = get_llm_client(
                openrouter_api_key=config.openrouter_api_key,
                model_name=config.openrouter_fallback_model,
            )
            response = fallback_llm.invoke([HumanMessage(content="Respond with 'READY' if you can read this message.")])
            reply = response.content.strip() if response.content else ""
            print(f"      ✓ Fallback model '{config.openrouter_fallback_model}' succeeded: {reply[:60]}")
            print("      [PASS] OpenRouter API fallback is operational!")
        except Exception as fb_err:
            print(f"      [FAIL] OpenRouter API failed with both primary and fallback models: {fb_err}")
            sys.exit(1)

    print("\n" + "=" * 65)
    print("   ALL CLOUD INFRASTRUCTURE VERIFICATIONS PASSED!")
    print("   Ready for live workshop demonstration.")
    print("=" * 65)


if __name__ == "__main__":
    run_cloud_validation()
