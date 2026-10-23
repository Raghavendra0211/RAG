# Learn RAG — AI-Powered PDF Research Assistant

A complete, production-grade **Retrieval-Augmented Generation (RAG)** application built in Python. This research assistant allows users to upload any digital PDF, indexes its contents into **Qdrant Cloud**, and allows users to ask natural-language questions that are answered with strict grounding and accurate, deduplicated source-page citations.

---

## 📑 Table of Contents

- [Project Overview](#project-overview)
- [How RAG Works & System Architecture](#how-rag-works--system-architecture)
- [Technology Stack](#technology-stack)
- [Prerequisites & Free Accounts](#prerequisites--free-accounts)
- [Installation & Setup (Windows PowerShell)](#installation--setup-windows-powershell)
- [Environment Configuration](#environment-configuration)
- [Running Automated Tests](#running-automated-tests)
- [Validating Live Cloud Infrastructure](#validating-live-cloud-infrastructure)
- [Launching the Streamlit Application](#launching-the-streamlit-application)
- [Workshop Demonstration Walkthrough](#workshop-demonstration-walkthrough)
- [Troubleshooting & FAQ](#troubleshooting--faq)

---

## 🎯 Project Overview

Large Language Models (LLMs) have a cutoff date and lack access to private, school, or organizational documents. Rather than fine-tuning an expensive model, **Retrieval-Augmented Generation (RAG)** bridges private documents and LLMs by:
1. Converting document text into numerical dense vectors (**embeddings**).
2. Storing those vectors in a high-speed vector database (**Qdrant Cloud**).
3. Finding the most semantically relevant passages when a user asks a question.
4. Supplying only those passages to an LLM (**OpenRouter**) to produce an answer backed by exact page numbers.

### Key Capabilities
- **Deterministic Document Identification:** Computes SHA-256 hashes of PDF bytes to prevent duplicate indexing and reuse existing cloud vectors.
- **Human-Friendly 1-Indexed Pages:** Normalizes raw 0-indexed page numbers to human-readable page 1...N.
- **Strict Grounding & Anti-Hallucination:** System prompt forbids outside assumptions. If the document does not contain the answer, it returns:
  > *"The provided document does not contain information to answer this question."*
- **Document Isolation:** Every similarity search is scoped strictly to the active `doc_id`. Content from different documents is never mixed.
- **Genuine Cosine Similarity Diagnostics:** An interactive panel exposes exact chunk numbers, source pages, character lengths, and cosine similarity scores.

---

## 🧠 How RAG Works & System Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                             THE RAG WORKFLOW                                │
└─────────────────────────────────────────────────────────────────────────────┘

[PHASE 1: DOCUMENT INGESTION & CLOUD INDEXING]
 1. Upload PDF ──► 2. PyPDFLoader ──► 3. RecursiveTextSplitter ──► 4. Embeddings ──► 5. Qdrant Cloud
   (Streamlit)      (Extract Pages     (1,000 chars size,           (all-MiniLM-L6      (learning_rag
                     + 1-idx Page)      400 chars overlap)           384 dimensions)     Collection)

[PHASE 2: RETRIEVAL & GROUNDED ANSWER GENERATION]
 6. User Query ──► 7. Vectorize Query ──► 8. Cosine Similarity ──► 9. OpenRouter LLM ──► 10. Answer +
   ("What is...")   (all-MiniLM-L6)          (Scoped to doc_id)     (recursive/flint)        Page Citations
```

### The Ingestion Pipeline
1. **Upload & Validation:** Validates that the uploaded PDF is not empty, corrupt, or an image-only scan without an OCR text layer.
2. **Page Extraction:** `PyPDFLoader` parses individual pages. Page numbers are normalized from 0-indexed to 1-indexed.
3. **Chunking:** `RecursiveCharacterTextSplitter` segments text into chunks of 1,000 characters with 400 characters of overlap.
4. **Dense Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` encodes each chunk into a 384-dimensional normalized vector.
5. **Qdrant Cloud Indexing:** Points are indexed in collection `learning_rag` with deterministic UUIDs and a payload index on `metadata.doc_id`.

### The Retrieval & Generation Pipeline
1. **Query Encoding:** The user's query is vectorized with the same embedding model.
2. **Document-Scoped Search:** Qdrant Cloud executes a cosine similarity search filtered strictly by `metadata.doc_id == active_doc_id`.
3. **Context Formation:** The top 4 retrieved chunks are assembled into labeled passages (`--- [Passage X | Source: ... | Page ...] ---`).
4. **Grounded Generation:** OpenRouter (`recursive/flint-v1.0:free`, with automatic fallback to `meta-llama/llama-3.1-8b-instruct`) answers strictly from the context.
5. **Citation Badges:** Deduplicated citations (`file.pdf (Pages 1, 3)`) and chunk inspection details are displayed.

---

## 🛠️ Technology Stack

| Component | Library / Service | Purpose |
|---|---|---|
| **Programming Language** | Python 3.10+ | Core language |
| **Web Interface** | `streamlit` | Clean SaaS dashboard and reactive chat UI |
| **PDF Extraction** | `pypdf`, `langchain-community` | Text extraction and page metadata preservation |
| **Text Chunking** | `langchain-text-splitters` | `RecursiveCharacterTextSplitter` (1000 chunk, 400 overlap) |
| **Embedding Model** | `sentence-transformers` | `all-MiniLM-L6-v2` (384 dimensions, normalized, CPU-friendly) |
| **Vector Database** | `qdrant-client`, `langchain-qdrant` | Managed Qdrant Cloud vector search |
| **LLM Gateway** | `langchain-openai` | OpenRouter API integration (`recursive/flint-v1.0:free`) |
| **Configuration** | `python-dotenv` | Secure `.env` environment variable loading |
| **Automated Testing** | `pytest` | Unit and pipeline test suite |

---

## 🔑 Prerequisites & Free Accounts

### 1. Python 3.10+
Download and install Python from [python.org](https://www.python.org/downloads/). Ensure **"Add Python to PATH"** is selected.

### 2. Qdrant Cloud (Free Tier)
1. Register for a free account at [https://cloud.qdrant.io/](https://cloud.qdrant.io/).
2. Create a Free Cluster (1 GB RAM, 0.5 vCPU).
3. Copy your **Cluster URL** (e.g., `https://your-cluster-id.us-east-1-0.aws.cloud.qdrant.io:6333`).
4. In **Data Access Control** > **API Keys**, create an API key and copy it.

### 3. OpenRouter (Free Account)
1. Sign up at [https://openrouter.ai/](https://openrouter.ai/).
2. Navigate to **Keys** ([https://openrouter.ai/keys](https://openrouter.ai/keys)) and click **Create Key**.
3. Copy the key (starts with `sk-or-v1-...`).
4. The project defaults to the free model `recursive/flint-v1.0:free`.

---

## 💻 Installation & Setup (Windows PowerShell)

Open **PowerShell** and navigate to the project directory:

```powershell
# Navigate into the project folder
cd C:\Users\PC-054\.gemini\antigravity-ide\scratch\learn-rag

# 1. Create a Python virtual environment
python -m venv venv

# 2. Activate the virtual environment
.\venv\Scripts\Activate.ps1
# If PowerShell policy prevents activation, run:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# and re-run .\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt
```

---

## ⚙️ Environment Configuration

1. Copy the `.env.example` template to `.env`:
   ```powershell
   Copy-Item .env.example .env
   ```
2. Open `.env` and insert your credentials:
   ```env
   # Qdrant Cloud Vector Database Configuration
   QDRANT_URL=https://your-actual-cluster-url.aws.cloud.qdrant.io:6333
   QDRANT_API_KEY=your_actual_qdrant_api_key
   QDRANT_COLLECTION_NAME=learning_rag

   # OpenRouter LLM Configuration
   OPENROUTER_API_KEY=sk-or-v1-your_actual_key_here
   OPENROUTER_MODEL=recursive/flint-v1.0:free
   OPENROUTER_FALLBACK_MODEL=meta-llama/llama-3.1-8b-instruct
   ```

> [!NOTE]
> Localhost URLs (such as `http://localhost:6333`) are explicitly disallowed. A cloud cluster is required per the workshop architecture.

---

## 🧪 Running Automated Tests

Run the complete automated unit test suite. These tests do **not** require live API keys or cloud credentials:

```powershell
pytest tests/test_rag_pipeline.py -v
```

### Test Coverage Summary:
- Configuration validation (missing keys, placeholder rejection, localhost rejection).
- Document hashing and duplicate detection.
- PDF extraction and 1-indexed human page numbers.
- Text splitting boundaries (1,000 char size, 400 overlap).
- Metadata preservation (`doc_id`, `filename`, `page`, `chunk_index`, `chunk_id`).
- Citation formatting and page deduplication.
- Empty retrieval handling.
- Strict grounding and exact refusal phrase verification.
- Document-scoped filtering isolation.

---

## 🌐 Validating Live Cloud Infrastructure

Once your `.env` file has real credentials, run the cloud validation tool:

```powershell
python tests/validate_cloud_setup.py
```

This script performs:
1. Qdrant Cloud connection & collection verification.
2. Safe test vector indexing & cosine similarity search.
3. Automatic cleanup of test vectors.
4. OpenRouter API ping and fallback check.

---

## 🚀 Launching the Streamlit Application

Start the web application:

```powershell
streamlit run app.py
```

If port `8501` is already in use by another service on your computer, launch on port `8502`:
```powershell
streamlit run app.py --server.port 8502
```

The application will open in your browser at `http://localhost:8501`.

---

## 🎓 Workshop Demonstration Walkthrough

When presenting this project to your workshop instructor:

1. **Show Setup & Health Status:**
   - Point out the sidebar: Qdrant Cloud and OpenRouter status indicators show active connections.
   - Expand the **About the RAG Architecture** panel to explain the 4-stage pipeline.

2. **Upload & Ingest a Document:**
   - Upload a PDF (e.g., lecture notes or `nodejs.pdf`).
   - Highlight the computed **Document ID** (SHA-256 hash).
   - Click **🚀 Index to Cloud** and walk through the 4 animated progress steps.

3. **Demonstrate Grounded Question-Answering:**
   - Ask a question contained in the document (e.g., *"What is Node.js?"*).
   - Show the answer generated by OpenRouter.
   - Point out the **Source Citations** badge (e.g., `nodejs.pdf (Pages 8, 10)`).

4. **Demonstrate Retrieval Diagnostics:**
   - Expand the **🔍 Inspect Retrieved Chunks** panel.
   - Show the 4 genuine cosine similarity scores (e.g. `0.6842`), original PDF page numbers, and exact chunk texts retrieved from Qdrant Cloud.

5. **Demonstrate Anti-Hallucination Refusal:**
   - Ask an unrelated question (e.g., *"What is the boiling point of liquid nitrogen?"*).
   - Show the system's exact refusal response:
     > *"The provided document does not contain information to answer this question."*
   - Note that no citations are fabricated.

6. **Demonstrate Document Isolation & Reuse:**
   - Refresh the page: the document is recognized as already indexed and can be used immediately without re-indexing.
   - Show the **🗑️ Delete Cloud Vectors** button which safely removes only this document's vectors from Qdrant Cloud.

---

## 🔧 Troubleshooting & FAQ

| Issue | Cause | Solution |
|---|---|---|
| `QDRANT_URL points to localhost` | Local URL used in `.env` | Local databases are not permitted. Use your Qdrant Cloud cluster URL (`https://...aws.cloud.qdrant.io:6333`). |
| `401 Unauthorized (Qdrant)` | Invalid or expired Qdrant API key | In Qdrant Cloud Console > **API Keys**, create a new key and update `QDRANT_API_KEY`. |
| `401 Unauthorized (OpenRouter)` | Missing key or invalid token | Check your key at [openrouter.ai/keys](https://openrouter.ai/keys). |
| `Model ... is not available (404)` | Provider updated free model IDs | The application automatically falls back to `meta-llama/llama-3.1-8b-instruct`. You can also configure `OPENROUTER_MODEL` in `.env`. |
| `PDF contains no extractable text` | PDF contains only scanned pictures | The document lacks digital text. Use a digital PDF or apply OCR first. |
| `ExecutionPolicy` error in PowerShell | Script execution disabled | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in your PowerShell window. |
