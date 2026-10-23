"""
Learn RAG — AI-Powered PDF Research Assistant
===============================================
A production-grade, grounded Question-Answering application built with Streamlit,
LangChain, Hugging Face embeddings, Qdrant Cloud, and OpenRouter.
Redesigned with a premium AI SaaS aesthetic inspired by Linear and Raycast.
"""

import os
import streamlit as st

from rag.config import get_config, validate_config, mask_secret
from rag.ingestion import (
    compute_document_id,
    load_pdf_from_bytes,
    split_document_pages,
    get_qdrant_client,
    ensure_collection_exists,
    is_document_indexed,
    delete_document_vectors,
    index_chunks_to_qdrant,
)
from rag.retrieval import (
    get_embedding_model,
    retrieve_relevant_chunks,
    format_citations,
)
from rag.generation import generate_grounded_answer, STANDARD_REFUSAL_MESSAGE


# Configure Streamlit page layout and appearance
st.set_page_config(
    page_title="Learn RAG — AI-Powered PDF Research Workspace",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# CUSTOM LUXURY SAAS STYLING (DARK PALETTE)
# ==========================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg-main: #080C14;
        --bg-surface: #0E1626;
        --bg-card: rgba(14, 22, 38, 0.7);
        --bg-card-hover: rgba(19, 30, 52, 0.85);
        --border-subtle: rgba(255, 255, 255, 0.08);
        --border-active: rgba(139, 92, 246, 0.35);
        --text-primary: #F8FAFC;
        --text-secondary: #94A3B8;
        --text-muted: #64748B;
        --violet-primary: #8B5CF6;
        --violet-deep: #7C3AED;
        --cyan-accent: #06B6D4;
        --cyan-light: #38BDF8;
        --emerald-glow: #10B981;
        --amber-glow: #F59E0B;
        --rose-glow: #EF4444;
    }

    /* Core Application Typography & Background */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    .stApp {
        background-color: var(--bg-main);
        background-image: 
            radial-gradient(ellipse 90% 60% at 50% -20%, rgba(124, 58, 237, 0.12), transparent),
            radial-gradient(circle at 100% 0%, rgba(6, 182, 212, 0.08), transparent 40%),
            radial-gradient(circle at 0% 100%, rgba(15, 23, 42, 0.7), transparent 50%);
        background-attachment: fixed;
        color: var(--text-primary);
    }

    /* Refined Custom Scrollbars */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(148, 163, 184, 0.2);
        border-radius: 9999px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(139, 92, 246, 0.45);
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0A0F1D !important;
        border-right: 1px solid var(--border-subtle);
    }
    section[data-testid="stSidebar"] > div:first-child {
        background-color: transparent;
    }

    /* Brand Header in Sidebar */
    .brand-container {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 4px 0 16px 0;
    }
    .brand-orb {
        width: 36px;
        height: 36px;
        border-radius: 10px;
        background: linear-gradient(135deg, #7C3AED 0%, #06B6D4 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 0 18px rgba(124, 58, 237, 0.45);
        color: #FFFFFF;
        font-size: 1.15rem;
        font-weight: 800;
    }
    .brand-title {
        font-size: 1.25rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #FFFFFF 30%, #C4B5FD 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }
    .brand-subtitle {
        font-size: 0.75rem;
        color: var(--text-muted);
        letter-spacing: 0.02em;
        font-weight: 500;
    }

    /* Navigation Radio Pills */
    div[data-testid="stRadio"] > div {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        padding: 4px;
        gap: 4px;
    }
    div[data-testid="stRadio"] label {
        border-radius: 8px;
        padding: 6px 12px;
        color: var(--text-secondary);
        font-weight: 500;
        font-size: 0.86rem;
        transition: all 0.2s ease;
    }
    div[data-testid="stRadio"] label:hover {
        color: var(--text-primary);
        background: rgba(255, 255, 255, 0.04);
    }

    /* Surface Card Base */
    .glass-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 14px;
        padding: 1.25rem;
        backdrop-filter: blur(14px);
        transition: transform 0.22s cubic-bezier(0.16, 1, 0.3, 1), 
                    border-color 0.22s ease, 
                    box-shadow 0.22s ease;
        position: relative;
        overflow: hidden;
    }
    .glass-card:hover {
        border-color: var(--border-active);
        box-shadow: 0 12px 30px -10px rgba(0, 0, 0, 0.5), 
                    0 0 20px -3px rgba(139, 92, 246, 0.12);
        transform: translateY(-2px);
    }

    /* Connection Status Dot with Pulse */
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
        position: relative;
    }
    .status-dot-emerald {
        background: var(--emerald-glow);
        box-shadow: 0 0 8px var(--emerald-glow);
    }
    .status-dot-rose {
        background: var(--rose-glow);
        box-shadow: 0 0 8px var(--rose-glow);
    }
    .status-dot-amber {
        background: var(--amber-glow);
        box-shadow: 0 0 8px var(--amber-glow);
    }

    /* Infrastructure Badge */
    .infra-badge {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: rgba(15, 23, 42, 0.55);
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        padding: 0.65rem 0.85rem;
        margin-bottom: 0.55rem;
        font-size: 0.82rem;
    }
    .infra-badge-title {
        color: var(--text-secondary);
        font-weight: 500;
        display: flex;
        align-items: center;
    }
    .infra-badge-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: #E2E8F0;
        background: rgba(255, 255, 255, 0.05);
        padding: 2px 7px;
        border-radius: 5px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        max-width: 140px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    /* Document Status Card */
    .doc-status-card {
        background: linear-gradient(145deg, rgba(20, 27, 45, 0.85) 0%, rgba(13, 20, 36, 0.65) 100%);
        border: 1px solid rgba(139, 92, 246, 0.22);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.85rem 0;
    }
    .doc-title {
        font-weight: 700;
        font-size: 0.94rem;
        color: #F8FAFC;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        margin-bottom: 4px;
    }
    .doc-meta {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: var(--text-muted);
        margin-bottom: 10px;
    }
    .stat-pill-group {
        display: flex;
        gap: 8px;
        margin-bottom: 8px;
    }
    .stat-pill {
        flex: 1;
        background: rgba(0, 0, 0, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 8px;
        padding: 6px 10px;
        text-align: center;
    }
    .stat-pill-num {
        font-size: 1.15rem;
        font-weight: 800;
        color: #A78BFA;
        line-height: 1.1;
    }
    .stat-pill-lbl {
        font-size: 0.68rem;
        text-transform: uppercase;
        color: var(--text-muted);
        letter-spacing: 0.05em;
        font-weight: 600;
        margin-top: 2px;
    }

    /* Hero Workspace Header */
    .workspace-header {
        padding: 0.5rem 0 1.25rem 0;
    }
    .pill-kicker {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        background: rgba(139, 92, 246, 0.12);
        border: 1px solid rgba(139, 92, 246, 0.3);
        color: #C4B5FD;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.75rem;
    }
    .hero-title {
        font-size: 2.35rem;
        font-weight: 800;
        letter-spacing: -0.035em;
        color: #FFFFFF;
        line-height: 1.15;
        margin: 0 0 0.45rem 0;
    }
    .hero-title-highlight {
        background: linear-gradient(135deg, #A78BFA 0%, #38BDF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-desc {
        font-size: 1rem;
        color: var(--text-secondary);
        font-weight: 400;
        max-width: 680px;
        line-height: 1.5;
        margin-bottom: 1.25rem;
    }

    /* Upload Hero Drop Card */
    .upload-hero-container {
        background: radial-gradient(circle at 50% 0%, rgba(124, 58, 237, 0.08) 0%, rgba(14, 22, 38, 0.6) 80%);
        border: 1px dashed rgba(139, 92, 246, 0.35);
        border-radius: 16px;
        padding: 1.75rem 1.5rem;
        text-align: center;
        margin-bottom: 1.5rem;
        transition: all 0.25s ease;
    }
    .upload-hero-container:hover {
        border-color: rgba(139, 92, 246, 0.6);
        box-shadow: 0 0 25px rgba(124, 58, 237, 0.12);
    }
    .upload-icon-wrapper {
        width: 54px;
        height: 54px;
        border-radius: 14px;
        background: rgba(139, 92, 246, 0.15);
        border: 1px solid rgba(139, 92, 246, 0.3);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 1.75rem;
        margin-bottom: 0.85rem;
        box-shadow: 0 0 15px rgba(124, 58, 237, 0.25);
    }
    .upload-headline {
        font-size: 1.15rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 0.35rem;
    }
    .upload-subheadline {
        font-size: 0.87rem;
        color: var(--text-muted);
        max-width: 480px;
        margin: 0 auto 0.75rem auto;
        line-height: 1.45;
    }
    .upload-specs {
        display: inline-flex;
        gap: 12px;
        font-size: 0.76rem;
        color: #CBD5E1;
        background: rgba(0, 0, 0, 0.25);
        padding: 4px 12px;
        border-radius: 9999px;
        border: 1px solid var(--border-subtle);
    }

    /* Suggested Question Chips */
    .chip-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 1rem 0;
    }
    .question-chip {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.09);
        color: #E2E8F0;
        padding: 0.45rem 0.9rem;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 500;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
        cursor: pointer;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .question-chip:hover {
        background: rgba(139, 92, 246, 0.2);
        border-color: rgba(139, 92, 246, 0.5);
        color: #FFFFFF;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(124, 58, 237, 0.25);
    }

    /* Citation Card Redesign */
    .citation-badge-card {
        background: rgba(14, 27, 46, 0.6);
        border: 1px solid rgba(6, 182, 212, 0.25);
        border-left: 3px solid var(--cyan-accent);
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin: 0.85rem 0;
        display: flex;
        align-items: center;
        gap: 10px;
        backdrop-filter: blur(8px);
    }
    .citation-icon {
        font-size: 1.1rem;
        color: var(--cyan-light);
    }
    .citation-text {
        font-size: 0.86rem;
        color: #E2E8F0;
        line-height: 1.4;
    }
    .citation-text strong {
        color: var(--cyan-light);
        font-weight: 600;
    }

    /* Diagnostic Retrieval Chunk Card */
    .retrieval-chunk-card {
        background: rgba(10, 16, 28, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.75rem;
        transition: all 0.2s ease;
    }
    .retrieval-chunk-card:hover {
        border-color: rgba(139, 92, 246, 0.3);
        background: rgba(15, 23, 42, 0.85);
    }
    .chunk-header-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
        flex-wrap: wrap;
        gap: 6px;
    }
    .chunk-index-badge {
        font-size: 0.75rem;
        font-weight: 700;
        color: #A78BFA;
        background: rgba(139, 92, 246, 0.15);
        border: 1px solid rgba(139, 92, 246, 0.25);
        padding: 2px 8px;
        border-radius: 6px;
    }
    .chunk-meta-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.74rem;
        color: var(--text-muted);
    }
    .similarity-pill {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--cyan-light);
        background: rgba(6, 182, 212, 0.12);
        border: 1px solid rgba(6, 182, 212, 0.25);
        padding: 2px 8px;
        border-radius: 6px;
    }

    /* Message Animation */
    div[data-testid="stChatMessage"] {
        animation: fadeRise 0.24s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        background: rgba(14, 22, 38, 0.55);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.85rem;
    }
    @keyframes fadeRise {
        from {
            opacity: 0;
            transform: translateY(6px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Primary Buttons & Interactive Overrides */
    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.88rem;
        padding: 0.5rem 1.15rem;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
        border: 1px solid var(--border-subtle);
    }
    div.stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    }
    button[kind="primary"] {
        background: linear-gradient(135deg, #7C3AED 0%, #4F46E5 100%) !important;
        border: 1px solid rgba(167, 139, 250, 0.4) !important;
        box-shadow: 0 4px 18px rgba(124, 58, 237, 0.35) !important;
        color: #FFFFFF !important;
    }
    button[kind="primary"]:hover {
        box-shadow: 0 6px 24px rgba(124, 58, 237, 0.55) !important;
        border-color: rgba(167, 139, 250, 0.7) !important;
    }

    /* Chat Input Styling */
    div[data-testid="stChatInput"] {
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        background: rgba(10, 16, 28, 0.85) !important;
        backdrop-filter: blur(16px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4) !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stChatInput"]:focus-within {
        border-color: rgba(139, 92, 246, 0.6) !important;
        box-shadow: 0 8px 32px rgba(124, 58, 237, 0.2) !important;
    }

    /* Respect accessibility motion reduction */
    @media (prefers-reduced-motion: reduce) {
        *, ::before, ::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "active_document" not in st.session_state:
    st.session_state.active_document = None

if "nav_view" not in st.session_state:
    st.session_state.nav_view = "Workspace"

if "is_indexing" not in st.session_state:
    st.session_state.is_indexing = False

if "confirm_delete" not in st.session_state:
    st.session_state.confirm_delete = False

# Load application configuration
config = get_config()
is_config_valid, config_issues = validate_config(config)

# ==========================================
# SIDEBAR: Infrastructure, Status & Controls
# ==========================================
with st.sidebar:
    # Brand Header
    st.markdown(
        """
        <div class="brand-container">
            <div class="brand-orb">⚡</div>
            <div>
                <div class="brand-title">Learn RAG</div>
                <div class="brand-subtitle">Your personal PDF intelligence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Navigation Switcher
    selected_view = st.radio(
        "Navigation",
        options=["Workspace", "Documents", "Retrieval Insights"],
        index=["Workspace", "Documents", "Retrieval Insights"].index(
            st.session_state.nav_view
            if st.session_state.nav_view in ["Workspace", "Documents", "Retrieval Insights"]
            else "Workspace"
        ),
        label_visibility="collapsed",
    )
    st.session_state.nav_view = selected_view

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Active Document Card
    st.markdown("<div style='font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; color: #94A3B8; font-weight: 700; margin-bottom: 6px;'>Active Document</div>", unsafe_allow_html=True)

    if st.session_state.active_document:
        doc = st.session_state.active_document
        doc_filename = doc.get("filename", "Untitled Document")
        doc_id = doc.get("doc_id", "N/A")
        total_pages = doc.get("total_pages", "—")
        total_chunks = doc.get("total_chunks", "—")

        st.markdown(
            f"""
            <div class="doc-status-card">
                <div class="doc-title" title="{doc_filename}">📄 {doc_filename}</div>
                <div class="doc-meta">ID: {doc_id[:12]}...{doc_id[-4:] if len(doc_id) > 16 else ''}</div>
                <div class="stat-pill-group">
                    <div class="stat-pill">
                        <div class="stat-pill-num">{total_pages}</div>
                        <div class="stat-pill-lbl">Pages</div>
                    </div>
                    <div class="stat-pill">
                        <div class="stat-pill-num">{total_chunks}</div>
                        <div class="stat-pill-lbl">Chunks</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_chg, col_del = st.columns(2)
        with col_chg:
            if st.button("🔄 Change", use_container_width=True, help="Switch to uploading or selecting a different PDF"):
                st.session_state.active_document = None
                st.session_state.confirm_delete = False
                st.rerun()

        with col_del:
            if not st.session_state.confirm_delete:
                if st.button("🗑️ Delete", use_container_width=True, help="Delete document vectors from Qdrant Cloud"):
                    st.session_state.confirm_delete = True
                    st.rerun()
            else:
                if st.button("⚠️ Confirm", use_container_width=True, type="primary", help="Permanently delete from Qdrant Cloud"):
                    with st.spinner("Deleting vectors from Qdrant Cloud..."):
                        try:
                            client = get_qdrant_client(config.qdrant_url, config.qdrant_api_key)
                            del_count = delete_document_vectors(
                                client, config.qdrant_collection_name, doc["doc_id"]
                            )
                            st.session_state.active_document = None
                            st.session_state.messages = []
                            st.session_state.confirm_delete = False
                            st.toast(f"Removed {del_count} vectors from Qdrant Cloud.", icon="🗑️")
                            st.rerun()
                        except Exception as err:
                            st.error(f"Failed to delete vectors: {err}")
                            st.session_state.confirm_delete = False
    else:
        st.markdown(
            """
            <div style="background: rgba(15, 23, 42, 0.45); border: 1px dashed rgba(255,255,255,0.1); border-radius: 10px; padding: 14px; text-align: center; margin-bottom: 12px;">
                <div style="font-size: 0.82rem; color: #94A3B8;">No active document</div>
                <div style="font-size: 0.74rem; color: #64748B; margin-top: 4px;">Upload a PDF in the workspace to begin</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Cloud Infrastructure Health Indicators
    st.markdown("<div style='font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; color: #94A3B8; font-weight: 700; margin: 16px 0 8px 0;'>Cloud Infrastructure</div>", unsafe_allow_html=True)

    # Genuine Qdrant Cloud Status
    if config.qdrant_url and "localhost" not in config.qdrant_url and config.qdrant_api_key:
        cluster_host = config.qdrant_url.split("//")[-1].split(":")[0]
        st.markdown(
            f"""
            <div class="infra-badge">
                <span class="infra-badge-title">
                    <span class="status-dot status-dot-emerald"></span>Qdrant Cloud
                </span>
                <span class="infra-badge-val" title="{cluster_host}">{cluster_host}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="infra-badge">
                <span class="infra-badge-title">
                    <span class="status-dot status-dot-rose"></span>Qdrant Cloud
                </span>
                <span class="infra-badge-val" style="color: #F87171;">Offline</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Genuine OpenRouter Status
    if config.openrouter_api_key:
        st.markdown(
            f"""
            <div class="infra-badge">
                <span class="infra-badge-title">
                    <span class="status-dot status-dot-emerald"></span>OpenRouter
                </span>
                <span class="infra-badge-val" title="{config.openrouter_model}">{config.openrouter_model.split('/')[-1]}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="infra-badge">
                <span class="infra-badge-title">
                    <span class="status-dot status-dot-rose"></span>OpenRouter
                </span>
                <span class="infra-badge-val" style="color: #F87171;">Key Missing</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Collection Badge
    st.markdown(
        f"""
        <div class="infra-badge">
            <span class="infra-badge-title">
                <span class="status-dot status-dot-emerald"></span>Collection
            </span>
            <span class="infra-badge-val" title="{config.qdrant_collection_name}">{config.qdrant_collection_name}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Diagnostics & History Actions
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    show_diagnostics = st.toggle(
        "Diagnostic Inspector",
        value=True,
        help="Expand retrieved chunk passages, cosine similarity scores, and page numbers with each answer."
    )

    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.toast("Chat history cleared.", icon="🧹")
        st.rerun()

    # Sidebar Footer
    st.markdown(
        """
        <div style="margin-top: 2.5rem; text-align: center; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 1rem;">
            <div style="font-size: 0.72rem; color: #64748B; font-weight: 500;">
                Learn RAG • v2.4.0
            </div>
            <div style="font-size: 0.68rem; color: #475569; margin-top: 2px;">
                Strict Grounding • Zero Hallucinations
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================
# CONFIGURATION GUARD
# ==========================================
if not is_config_valid:
    st.markdown(
        """
        <div class="workspace-header">
            <div class="pill-kicker">Configuration Required</div>
            <h1 class="hero-title">Environment Setup Incomplete</h1>
            <p class="hero-desc">
                Learn RAG requires active credentials for <strong>Qdrant Cloud</strong> and <strong>OpenRouter</strong>.
                Please populate your <code>.env</code> file based on <code>.env.example</code>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("📋 Review Configuration Checklist", expanded=True):
        for issue in config_issues:
            st.error(f"❌ {issue}")
        st.code(
            """# .env template
QDRANT_URL=https://your-cluster-id.us-east-1-0.aws.cloud.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key_here
QDRANT_COLLECTION_NAME=learning_rag
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=recursive/flint-v1.0:free
OPENROUTER_FALLBACK_MODEL=meta-llama/llama-3.1-8b-instruct
""",
            language="env",
        )
    st.stop()


# =======================================================
# VIEW 1: WORKSPACE (Main Document Upload & Grounded QA)
# =======================================================
if st.session_state.nav_view == "Workspace":
    # Workspace Header
    st.markdown(
        """
        <div class="workspace-header">
            <div class="pill-kicker">
                <span class="status-dot status-dot-emerald" style="margin: 0;"></span>
                AI Research Workspace • Grounded RAG
            </div>
            <h1 class="hero-title">Chat with your <span class="hero-title-highlight">documents.</span></h1>
            <p class="hero-desc">
                Ask deep questions grounded strictly in your uploaded PDF. Every claim is verified against dense vector passages with exact page citations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------
    # Document Upload & Ingestion Section
    # ------------------------------------------
    # Show upload card if no document is active, or an expander if one is active
    show_upload_card = st.session_state.active_document is None

    if show_upload_card:
        st.markdown(
            """
            <div class="upload-hero-container">
                <div class="upload-icon-wrapper">📂</div>
                <div class="upload-headline">Drop your knowledge here</div>
                <div class="upload-subheadline">
                    Upload research papers, lecture notes, or technical documentation to build an isolated, searchable vector index in Qdrant Cloud.
                </div>
                <div class="upload-specs">
                    <span>📄 PDF Documents Only</span>
                    <span>•</span>
                    <span>⚡ Extractable Digital Text</span>
                    <span>•</span>
                    <span>🔒 Scoped Document Isolation</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with (st.container() if show_upload_card else st.expander("📤 Upload / Index a Different Document", expanded=False)):
        uploaded_file = st.file_uploader(
            "Select or drop a PDF file:",
            type=["pdf"],
            help="Upload a PDF with selectable digital text. Scanned or image-only PDFs require OCR preprocessing.",
            key="pdf_uploader",
        )

        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            filename = uploaded_file.name

            if len(file_bytes) == 0:
                st.error(f"❌ The file `{filename}` is empty (0 bytes). Please upload a valid PDF.")
            else:
                doc_id = compute_document_id(file_bytes, filename)
                file_kb = len(file_bytes) / 1024

                # Check if already indexed in Qdrant Cloud
                already_indexed = False
                existing_chunks = 0
                try:
                    qdrant_client = get_qdrant_client(config.qdrant_url, config.qdrant_api_key)
                    already_indexed, existing_chunks = is_document_indexed(
                        qdrant_client, config.qdrant_collection_name, doc_id
                    )
                except Exception as conn_err:
                    st.error(
                        f"❌ Failed to reach Qdrant Cloud: {conn_err}. "
                        "Verify your `QDRANT_URL` and `QDRANT_API_KEY`."
                    )

                st.markdown(
                    f"""
                    <div class="glass-card" style="margin: 1rem 0;">
                        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                            <div>
                                <div style="font-weight: 700; font-size: 1.05rem; color: #FFFFFF;">{filename}</div>
                                <div style="font-size: 0.8rem; color: #94A3B8; font-family: 'JetBrains Mono', monospace; margin-top: 3px;">
                                    Size: {file_kb:.1f} KB • Doc ID: {doc_id[:16]}...
                                </div>
                            </div>
                            <div>
                                <span class="similarity-pill" style="font-size: 0.8rem;">
                                    {"✅ Indexed in Cloud (" + str(existing_chunks) + " vectors)" if already_indexed else "⏳ Ready to Index"}
                                </span>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                col_action_left, col_action_right = st.columns([1, 1])

                with col_action_left:
                    if already_indexed:
                        if st.button("👉 Use this indexed document for chat", use_container_width=True, type="primary"):
                            st.session_state.active_document = {
                                "doc_id": doc_id,
                                "filename": filename,
                                "total_chunks": existing_chunks,
                            }
                            st.rerun()
                    else:
                        btn_label = "🚀 Index to Qdrant Cloud"
                        start_indexing = st.button(
                            btn_label,
                            use_container_width=True,
                            type="primary",
                            disabled=st.session_state.is_indexing,
                        )

                with col_action_right:
                    if already_indexed:
                        start_indexing = st.button(
                            "🔄 Re-Index from Scratch",
                            use_container_width=True,
                            disabled=st.session_state.is_indexing,
                        )
                    else:
                        start_indexing = False

                # Ingestion execution
                if start_indexing and not st.session_state.is_indexing:
                    st.session_state.is_indexing = True
                    with st.status("Executing Grounded RAG Ingestion Pipeline...", expanded=True) as status:
                        try:
                            # Step 1: PDF Extraction & 1-indexed page conversion
                            status.write("📄 **Step 1/4:** Parsing PDF pages and preserving 1-indexed numbers...")
                            pages = load_pdf_from_bytes(file_bytes, filename)
                            status.write(f"✓ Extracted {len(pages)} page(s) with clean digital text.")

                            # Step 2: Recursive Character Text Splitting
                            status.write(
                                f"✂️ **Step 2/4:** Splitting text into dense chunks (Size: {config.chunk_size}, Overlap: {config.chunk_overlap} chars)..."
                            )
                            chunks = split_document_pages(
                                documents=pages,
                                doc_id=doc_id,
                                filename=filename,
                                chunk_size=config.chunk_size,
                                chunk_overlap=config.chunk_overlap,
                            )
                            status.write(f"✓ Generated {len(chunks)} enriched text chunks.")

                            # Step 3: Embeddings Initialization
                            status.write(
                                f"🧠 **Step 3/4:** Initializing embeddings model (`{config.embedding_model_name}`)..."
                            )
                            embeddings = get_embedding_model(config.embedding_model_name)
                            status.write("✓ Embeddings model ready (384-dimensional dense vectors).")

                            # Step 4: Index into Qdrant Cloud
                            status.write(
                                f"☁️ **Step 4/4:** Indexing vectors & metadata into Qdrant Cloud collection `{config.qdrant_collection_name}`..."
                            )
                            indexed_count = index_chunks_to_qdrant(
                                client=qdrant_client,
                                embeddings_model=embeddings,
                                collection_name=config.qdrant_collection_name,
                                chunks=chunks,
                                doc_id=doc_id,
                                overwrite_existing=True,
                            )

                            st.session_state.active_document = {
                                "doc_id": doc_id,
                                "filename": filename,
                                "total_pages": len(pages),
                                "total_chunks": indexed_count,
                            }
                            st.session_state.is_indexing = False

                            status.update(
                                label=f"🎉 Successfully indexed '{filename}' ({indexed_count} chunks) in Qdrant Cloud!",
                                state="complete",
                                expanded=False,
                            )
                            st.toast(f"'{filename}' ready for research!", icon="🚀")
                            st.rerun()

                        except Exception as exc:
                            st.session_state.is_indexing = False
                            status.update(label=f"❌ Ingestion failed: {exc}", state="error")
                            st.error(f"Ingestion Error: {exc}")

    # ------------------------------------------
    # AI Conversation Workspace
    # ------------------------------------------
    if not st.session_state.active_document:
        st.markdown(
            """
            <div style="background: rgba(15, 23, 42, 0.4); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 14px; padding: 2.5rem 1.5rem; text-align: center; margin-top: 1.5rem;">
                <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">📚</div>
                <div style="font-weight: 700; font-size: 1.1rem; color: #F8FAFC;">Document Knowledge Base Awaiting Input</div>
                <div style="font-size: 0.88rem; color: #94A3B8; max-width: 440px; margin: 0.4rem auto 0 auto; line-height: 1.5;">
                    Upload or select an indexed document above to begin asking grounded research questions.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        active_doc = st.session_state.active_document

        # Active Document Status Header Bar
        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: center; background: rgba(14, 22, 38, 0.65); border: 1px solid rgba(139, 92, 246, 0.25); border-radius: 12px; padding: 0.75rem 1.15rem; margin: 1.25rem 0; flex-wrap: wrap; gap: 8px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 1.2rem;">📑</span>
                    <div>
                        <span style="font-weight: 700; font-size: 0.95rem; color: #FFFFFF;">{active_doc['filename']}</span>
                        <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.76rem; color: #94A3B8; margin-left: 8px;">
                            (Scoped ID: {active_doc['doc_id'][:12]}...)
                        </span>
                    </div>
                </div>
                <div style="display: flex; gap: 8px; align-items: center;">
                    <span class="similarity-pill">Qdrant Cloud: {config.qdrant_collection_name}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Suggested Questions if Chat History is Empty
        if len(st.session_state.messages) == 0:
            st.markdown(
                """
                <div style="margin: 1.5rem 0 0.75rem 0;">
                    <div style="font-size: 0.82rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #A78BFA;">
                        💡 Suggested Explorations
                    </div>
                    <div style="font-size: 0.84rem; color: #94A3B8; margin-top: 2px;">
                        Click any prompt below to query your document directly:
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            sample_prompts = [
                "Summarize the key concepts and takeaways from this document.",
                "What are the main architectures, components, or methodologies described?",
                "List any technical prerequisites, dependencies, or key rules mentioned.",
            ]

            cols = st.columns(len(sample_prompts))
            for idx, prompt_text in enumerate(sample_prompts):
                with cols[idx]:
                    if st.button(prompt_text, key=f"prompt_chip_{idx}", use_container_width=True):
                        st.session_state.pending_query = prompt_text
                        st.rerun()

        # Render Conversation Messages
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

                if msg["role"] == "assistant":
                    citations = msg.get("citations", {})
                    citation_summary = citations.get("citation_summary", "")

                    # Render grounded source citation badge if present
                    if citation_summary and "No citations" not in citation_summary:
                        st.markdown(
                            f"""
                            <div class="citation-badge-card">
                                <span class="citation-icon">📄</span>
                                <div class="citation-text">
                                    <strong>Source Citations:</strong> {citation_summary}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # Render expandable retrieval inspector
                    chunks = msg.get("chunks", [])
                    if chunks:
                        with st.expander(
                            f"🔍 Retrieval Inspector ({len(chunks)} passages from Qdrant Cloud)",
                            expanded=show_diagnostics,
                        ):
                            for i, ch in enumerate(chunks, 1):
                                pg = ch.get("page", 1)
                                score = ch.get("score")
                                score_str = f"{score:.4f}" if score is not None else "N/A"
                                char_len = len(ch.get("content", ""))
                                src_file = ch.get("filename", active_doc["filename"])
                                
                                st.markdown(
                                    f"""
                                    <div class="retrieval-chunk-card">
                                        <div class="chunk-header-row">
                                            <div>
                                                <span class="chunk-index-badge">Passage #{i}</span>
                                                <span class="chunk-meta-tag" style="margin-left: 8px;">Source: <strong>{src_file}</strong> (Page {pg})</span>
                                            </div>
                                            <div>
                                                <span class="similarity-pill">Score: {score_str}</span>
                                                <span class="chunk-meta-tag" style="margin-left: 6px;">{char_len} chars</span>
                                            </div>
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                                st.code(ch.get("content", ""), language="text")

        # Chat Input Bar
        # Check if a pending prompt was triggered via chip
        default_query = None
        if "pending_query" in st.session_state and st.session_state.pending_query:
            default_query = st.session_state.pop("pending_query")

        user_query = st.chat_input("Ask a grounded question about your PDF document...")
        query_to_execute = default_query if default_query else user_query

        if query_to_execute:
            st.session_state.messages.append({"role": "user", "content": query_to_execute})
            with st.chat_message("user"):
                st.markdown(query_to_execute)

            with st.chat_message("assistant"):
                with st.spinner("🔍 Retrieving dense vector passages & synthesizing answer..."):
                    try:
                        qdrant_client = get_qdrant_client(config.qdrant_url, config.qdrant_api_key)
                        embeddings = get_embedding_model(config.embedding_model_name)

                        # Retrieve top relevant chunks filtered strictly by active document ID
                        retrieved_docs = retrieve_relevant_chunks(
                            client=qdrant_client,
                            embeddings_model=embeddings,
                            collection_name=config.qdrant_collection_name,
                            query=query_to_execute,
                            doc_id=active_doc["doc_id"],
                            top_k=config.top_k,
                        )

                        # Generate strictly grounded answer via OpenRouter
                        result = generate_grounded_answer(
                            query=query_to_execute,
                            retrieved_chunks=retrieved_docs,
                            openrouter_api_key=config.openrouter_api_key,
                            model_name=config.openrouter_model,
                            fallback_model=config.openrouter_fallback_model,
                        )

                        answer_text = result["answer"]
                        citations = result["citations"]
                        citation_summary = citations.get("citation_summary", "")

                        st.markdown(answer_text)

                        # Display source citations if grounded
                        if result.get("grounded", False) and citation_summary and "No citations" not in citation_summary:
                            st.markdown(
                                f"""
                                <div class="citation-badge-card">
                                    <span class="citation-icon">📄</span>
                                    <div class="citation-text">
                                        <strong>Source Citations:</strong> {citation_summary}
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        # Format diagnostic chunk details
                        chunk_details = [
                            {
                                "filename": doc.metadata.get("filename", active_doc["filename"]),
                                "page": doc.metadata.get("page", 1),
                                "score": doc.metadata.get("similarity_score"),
                                "content": doc.page_content,
                            }
                            for doc in retrieved_docs
                        ]

                        if chunk_details:
                            with st.expander(
                                f"🔍 Retrieval Inspector ({len(chunk_details)} passages from Qdrant Cloud)",
                                expanded=show_diagnostics,
                            ):
                                for i, ch in enumerate(chunk_details, 1):
                                    pg = ch["page"]
                                    sc = ch["score"]
                                    sc_str = f"{sc:.4f}" if sc is not None else "N/A"
                                    char_len = len(ch["content"])
                                    src_file = ch["filename"]

                                    st.markdown(
                                        f"""
                                        <div class="retrieval-chunk-card">
                                            <div class="chunk-header-row">
                                                <div>
                                                    <span class="chunk-index-badge">Passage #{i}</span>
                                                    <span class="chunk-meta-tag" style="margin-left: 8px;">Source: <strong>{src_file}</strong> (Page {pg})</span>
                                                </div>
                                                <div>
                                                    <span class="similarity-pill">Score: {sc_str}</span>
                                                    <span class="chunk-meta-tag" style="margin-left: 6px;">{char_len} chars</span>
                                                </div>
                                            </div>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )
                                    st.code(ch["content"], language="text")

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": answer_text,
                                "citations": citations if result.get("grounded", False) else {},
                                "chunks": chunk_details,
                                "model": result.get("model_used"),
                            }
                        )

                    except Exception as query_exc:
                        err_text = f"❌ Error retrieving answer: {query_exc}"
                        st.error(err_text)
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": err_text,
                                "citations": {},
                                "chunks": [],
                            }
                        )


# =======================================================
# VIEW 2: DOCUMENTS (Document Vault & Vector Analytics)
# =======================================================
elif st.session_state.nav_view == "Documents":
    st.markdown(
        """
        <div class="workspace-header">
            <div class="pill-kicker">
                <span class="status-dot status-dot-emerald" style="margin: 0;"></span>
                Document Vault
            </div>
            <h1 class="hero-title">Document & Index <span class="hero-title-highlight">Analytics.</span></h1>
            <p class="hero-desc">
                Inspect vector storage metrics, page segmentations, and collection schema for your active document.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.active_document:
        st.info("ℹ️ No document is currently active. Upload or select a PDF in the Workspace.")
    else:
        active_doc = st.session_state.active_document

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                f"""
                <div class="glass-card" style="text-align: center;">
                    <div style="font-size: 0.76rem; text-transform: uppercase; color: #94A3B8; font-weight: 700;">Total Pages</div>
                    <div style="font-size: 1.8rem; font-weight: 800; color: #A78BFA; margin-top: 4px;">{active_doc.get('total_pages', 'N/A')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""
                <div class="glass-card" style="text-align: center;">
                    <div style="font-size: 0.76rem; text-transform: uppercase; color: #94A3B8; font-weight: 700;">Vector Chunks</div>
                    <div style="font-size: 1.8rem; font-weight: 800; color: #38BDF8; margin-top: 4px;">{active_doc.get('total_chunks', 'N/A')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f"""
                <div class="glass-card" style="text-align: center;">
                    <div style="font-size: 0.76rem; text-transform: uppercase; color: #94A3B8; font-weight: 700;">Chunk Size</div>
                    <div style="font-size: 1.8rem; font-weight: 800; color: #10B981; margin-top: 4px;">{config.chunk_size}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                f"""
                <div class="glass-card" style="text-align: center;">
                    <div style="font-size: 0.76rem; text-transform: uppercase; color: #94A3B8; font-weight: 700;">Chunk Overlap</div>
                    <div style="font-size: 1.8rem; font-weight: 800; color: #F59E0B; margin-top: 4px;">{config.chunk_overlap}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="glass-card">
                <h3 style="font-size: 1.15rem; font-weight: 700; color: #FFFFFF; margin-bottom: 0.85rem;">Document Identification & Schema</h3>
                <table style="width: 100%; border-collapse: collapse; font-size: 0.88rem;">
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.06);">
                        <td style="padding: 10px 0; color: #94A3B8; width: 35%;">File Name</td>
                        <td style="padding: 10px 0; font-weight: 600; color: #FFFFFF;">{active_doc.get('filename')}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.06);">
                        <td style="padding: 10px 0; color: #94A3B8;">Document Hash (doc_id)</td>
                        <td style="padding: 10px 0; font-family: 'JetBrains Mono', monospace; color: #C4B5FD;">{active_doc.get('doc_id')}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.06);">
                        <td style="padding: 10px 0; color: #94A3B8;">Embedding Model</td>
                        <td style="padding: 10px 0; font-family: 'JetBrains Mono', monospace; color: #38BDF8;">{config.embedding_model_name} (384-dim, normalized)</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.06);">
                        <td style="padding: 10px 0; color: #94A3B8;">Target Qdrant Collection</td>
                        <td style="padding: 10px 0; font-family: 'JetBrains Mono', monospace; color: #10B981;">{config.qdrant_collection_name}</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px 0; color: #94A3B8;">Isolation Filter Key</td>
                        <td style="padding: 10px 0; font-family: 'JetBrains Mono', monospace; color: #F59E0B;">payload.doc_id == "{active_doc.get('doc_id')}"</td>
                    </tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =======================================================
# VIEW 3: RETRIEVAL INSIGHTS (Inspector & Analytics)
# =======================================================
elif st.session_state.nav_view == "Retrieval Insights":
    st.markdown(
        """
        <div class="workspace-header">
            <div class="pill-kicker">
                <span class="status-dot status-dot-emerald" style="margin: 0;"></span>
                Retrieval Insights
            </div>
            <h1 class="hero-title">Vector Search <span class="hero-title-highlight">Inspector.</span></h1>
            <p class="hero-desc">
                Review retrieved passages, cosine similarity score distributions, and citation mappings across recent queries.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    assistant_msgs = [m for m in st.session_state.messages if m.get("role") == "assistant" and m.get("chunks")]

    if not assistant_msgs:
        st.info("💡 No queries have been executed yet in this session. Ask a question in the Workspace to populate retrieval insights.")
    else:
        st.markdown(f"### Recent Queries ({len(assistant_msgs)})")

        for idx, msg in enumerate(reversed(assistant_msgs), 1):
            chunks = msg.get("chunks", [])
            citations = msg.get("citations", {})
            summary = citations.get("citation_summary", "No citations")

            with st.expander(f"🔎 Query #{len(assistant_msgs) - idx + 1} • {len(chunks)} Chunks Retrieved • Citations: {summary}", expanded=(idx == 1)):
                st.markdown(f"**Answer Summary:** {msg.get('content', '')[:180]}...")
                st.markdown("---")

                for ch_i, ch in enumerate(chunks, 1):
                    pg = ch.get("page", 1)
                    score = ch.get("score")
                    score_str = f"{score:.4f}" if score is not None else "N/A"
                    char_len = len(ch.get("content", ""))

                    st.markdown(
                        f"""
                        <div class="retrieval-chunk-card">
                            <div class="chunk-header-row">
                                <div>
                                    <span class="chunk-index-badge">Chunk #{ch_i}</span>
                                    <span class="chunk-meta-tag" style="margin-left: 8px;">Page {pg} • Length: {char_len} chars</span>
                                </div>
                                <div>
                                    <span class="similarity-pill">Cosine Similarity: {score_str}</span>
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.code(ch.get("content", ""), language="text")
