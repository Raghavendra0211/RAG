# Complete Step-by-Step Implementation Guide: Building a Cloud-Native RAG Application

This guide provides a comprehensive, step-by-step walkthrough for building a production-grade, beginner-friendly **Retrieval-Augmented Generation (RAG)** application from scratch. 

The application allows users to upload any text-based PDF, index its contents into a managed **Qdrant Cloud** vector database, and ask natural language questions. Answers are generated using **OpenRouter** (`recursive/flint-v1.0:free`), grounded **exclusively** in the uploaded document, and accompanied by **exact page-number citations** and an interactive chunk inspection panel.

---

## 1. Project Overview & How RAG Works

Large Language Models (LLMs) are trained on massive public datasets, but they do not know about private, recent, or domain-specific documents (such as your school syllabus, research papers, or company handbooks). Fine-tuning an LLM on every document is expensive, slow, and prone to hallucinations.

**Retrieval-Augmented Generation (RAG)** solves this by retrieving relevant text excerpts from a document at query time and injecting them into the LLM's prompt as trusted context.

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                             THE RAG WORKFLOW                                │
└─────────────────────────────────────────────────────────────────────────────┘

[PHASE A: DOCUMENT INGESTION & CLOUD INDEXING]
 1. Upload PDF ──► 2. PyPDFLoader ──► 3. RecursiveTextSplitter ──► 4. Embeddings ──► 5. Qdrant Cloud
   (Streamlit)      (Extract Pages     (1,000 chars size,           (all-MiniLM-L6      (learning_rag
                     + 1-idx Page)      400 chars overlap)           384 dimensions)     Collection)

[PHASE B: RETRIEVAL & GROUNDED ANSWER GENERATION]
 6. User Query ──► 7. Vectorize Query ──► 8. Cosine Similarity ──► 9. OpenRouter LLM ──► 10. Answer +
   ("What is...")   (all-MiniLM-L6)          (Scoped to doc_id)     (recursive/flint)        Page Citations
```

### The 4 Core Stages Explained:
1. **Ingestion & Page Extraction:** Read the PDF page-by-page. LangChain's `PyPDFLoader` numbers pages starting at `0` (page 0 is page 1). We convert this to human-friendly 1-indexed numbers (`page + 1`) and compute a unique SHA-256 hash (`doc_id`) of the file bytes.
2. **Chunking:** Documents are too large to embed as single blocks. We split pages into overlapping chunks of **1,000 characters** with **400 characters of overlap** using `RecursiveCharacterTextSplitter`. Overlap ensures that sentences split across chunk boundaries do not lose their semantic context.
3. **Embedding Generation:** Each text chunk is converted into a 384-dimensional dense numerical vector using Hugging Face's `sentence-transformers/all-MiniLM-L6-v2`. Semantic meaning is captured in geometric space: chunks with similar meanings point in similar directions.
4. **Cloud Storage & Filtered Retrieval:** Vectors and metadata (`doc_id`, `filename`, `page`, `chunk_index`) are stored in **Qdrant Cloud**. When a question is asked, it is converted into a vector, and Qdrant computes **Cosine Similarity** to find the top matching chunks. The search is filtered by `doc_id` so that multiple uploaded documents never contaminate each other.
5. **Grounded Answer Generation:** The retrieved passages are inserted into a prompt sent to OpenRouter (`recursive/flint-v1.0:free`). A strict grounding prompt instructs the model to answer using *only* the provided excerpts. If the information is missing, the model responds with: `"The provided document does not contain information to answer this question."`

---

## 2. Technology Stack & Component Roles

| Technology | Role | Why It Was Chosen |
|---|---|---|
| **Python 3.10+** | Core programming language | Industry standard for AI, LangChain, and data science |
| **Streamlit (`streamlit`)** | Web application framework | Enables rapid creation of reactive, interactive web UIs in pure Python |
| **LangChain Core (`langchain-core`)** | Document & prompt abstractions | Provides standardized schemas for `Document`, prompts, and messages |
| **PyPDFLoader (`langchain-community`, `pypdf`)** | PDF text extraction | Extracts clean digital text and preserves per-page metadata |
| **RecursiveCharacterTextSplitter** | Text chunking | Splits text respecting paragraph, sentence, and word boundaries |
| **Hugging Face (`sentence-transformers`)** | Local embedding model | `all-MiniLM-L6-v2` runs fast on CPU, produces 384-dim normalized vectors |
| **Qdrant Cloud (`qdrant-client`, `langchain-qdrant`)** | Cloud vector database | Managed, high-speed vector index with metadata filtering. **No Docker needed.** |
| **OpenRouter (`langchain-openai`)** | LLM API gateway | Provides OpenAI-compatible endpoint for models like `recursive/flint-v1.0:free` |
| **python-dotenv** | Secret management | Loads API keys from `.env` without hardcoding credentials |
| **pytest** | Automated test suite | Verifies chunking, metadata preservation, and prompt grounding |

---

## 3. Prerequisites & Required Accounts

Before writing code, set up the following free accounts:

### 1. Python 3.10+
- Download and install Python from [python.org](https://www.python.org/downloads/).
- During installation on Windows, ensure you check **"Add Python to PATH"**.

### 2. Qdrant Cloud Account (Free Tier)
1. Sign up at [https://cloud.qdrant.io/](https://cloud.qdrant.io/).
2. Click **Create Cluster**, choose the **Free Tier (1 GB RAM / 0.5 vCPU)**, and select a region (e.g., AWS `us-east-1` or `eu-west-2`).
3. Once created, copy the **Cluster URL** (e.g., `https://your-cluster-id.region.aws.cloud.qdrant.io:6333` or `https://your-cluster-id.region.aws.cloud.qdrant.io`).
4. Click **Data Access Control** > **API Keys** > **Create API Key**. Copy your API key.

### 3. OpenRouter Account
1. Sign up at [https://openrouter.ai/](https://openrouter.ai/).
2. Go to **Keys** ([https://openrouter.ai/keys](https://openrouter.ai/keys)) and click **Create Key**.
3. Name it `learn-rag-workshop` and copy the generated secret key (starts with `sk-or-v1-...`).
4. The project uses the free-tier model `recursive/flint-v1.0:free`, so credit purchase is not required to get started.

---

## 4. Project Folder Structure & File Explanations

Organize your workspace as follows:

```text
learn-rag/
├── app.py                      # Streamlit UI & interactive chat workflow
├── rag/                        # Core modular RAG package
│   ├── __init__.py             # Public package exports
│   ├── config.py               # Environment configuration and validation
│   ├── ingestion.py            # PDF loading, SHA-256 hashing, chunking, Qdrant upload
│   ├── retrieval.py            # Embedding cache, scoped similarity search, citations
│   └── generation.py           # Grounded prompt engineering and OpenRouter LLM call
├── tests/                      # Automated test suite
│   ├── __init__.py             # Test package initializer
│   ├── test_rag_pipeline.py    # Unit tests for chunking, citations, and grounding
│   └── validate_cloud_setup.py # End-to-end live cloud infrastructure test
├── requirements.txt            # Python package dependencies
├── .env.example                # Template for environment configuration
├── .gitignore                  # Git ignore rules for secrets and virtualenvs
├── README.md                   # Project summary and workshop guide
└── implementation.md           # This comprehensive recreation guide
```

---

## 5. Python Virtual Environment & Dependency Setup

Run the following commands in **Windows PowerShell**:

### Step 5.1: Create & Activate Virtual Environment
```powershell
# Open terminal in project directory
python -m venv venv

# Activate virtual environment on Windows
.\venv\Scripts\Activate.ps1
```

*(If you receive an execution policy restriction error in PowerShell, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and re-run the activate command).*

### Step 5.2: Install Requirements
```powershell
pip install -r requirements.txt
```

---

## 6. Testing & Workshop Verification Guide

To test the application in a classroom or technical workshop:

### Test Case 1: Testing Supported Content (e.g. Node.js PDF)
1. Upload a sample PDF (e.g., `nodejs.pdf`).
2. Click **🚀 Index to Cloud** and observe the progress bars.
3. In the chat box, ask:
   ```text
   What is Node.js?
   ```
4. **Expected Behavior:**
   - The model responds explaining that Node.js explores the V8 JavaScript engine, non-blocking I/O, and is commonly used as a web server.
   - The **Source Citations** badge displays: `nodejs.pdf (Pages 8, 10, 23)`.
   - The **Inspect Retrieved Chunks** panel shows the exact four passages retrieved from Qdrant Cloud along with their cosine similarity scores.

### Test Case 2: Testing Feature-Specific Question
1. Ask:
   ```text
   What does the Node.js module system let you do?
   ```
2. **Expected Behavior:**
   - The model answers that the module system enables loading external libraries, built-in core modules, and third-party npm modules.
   - Citations display: `nodejs.pdf (Pages 8, 10)`.

### Test Case 3: Testing Anti-Hallucination & Refusal
1. Ask an unrelated question not mentioned anywhere in the document:
   ```text
   What is the speed of light in vacuum?
   ```
2. **Expected Behavior:**
   - The model does **not** hallucinate or provide general physics knowledge.
   - It strictly replies:
     > *"The provided document does not contain information to answer this question."*

---

## 7. Common Errors & Troubleshooting

| Error / Symptom | Root Cause | Solution |
|---|---|---|
| `QDRANT_URL points to localhost` | Local Docker address used in `.env` | Local databases are not permitted. Use your official Qdrant Cloud cluster URL (`https://...aws.cloud.qdrant.io:6333`). |
| `401 Unauthorized (Qdrant)` | Invalid or expired Qdrant API key | Navigate to Qdrant Cloud Console > **API Keys**, generate a new key, and update `QDRANT_API_KEY` in `.env`. |
| `401 Unauthorized (OpenRouter)` | Missing key or insufficient credits | Verify your key at [openrouter.ai/keys](https://openrouter.ai/keys). Ensure `OPENROUTER_MODEL=recursive/flint-v1.0:free`. |
| `Model ... is no longer available on OpenRouter (404)` | Provider retired the model identifier | The application includes an automatic fallback to `meta-llama/llama-3.1-8b-instruct`. You can also configure any active OpenRouter model in `.env`. |
| `PDF contains no extractable text` | PDF contains scanned images only | `PyPDFLoader` requires selectable digital text. Use a digital PDF or apply an OCR tool beforehand. |
| Model refuses valid questions (`Does not contain info...`) | Greedy decoding over-refusal at `temp=0.0` | Ensure `temperature=0.2` in `rag/generation.py` to allow natural synthesis while strictly bounding output to context. |
| Port 8501 is not available | Another instance of Streamlit is running | Start Streamlit on another port: `streamlit run app.py --server.port 8502`. |
