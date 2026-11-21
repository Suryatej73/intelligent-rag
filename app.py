import os
import streamlit as st
from pathlib import Path

import config
from rag_chain import ask_bot, get_llm
from vectorstore import build_or_load_vectorstore, get_retriever

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Zyro Dynamics | HR Intelligence Help Desk",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom Styling ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* App Background */
.stApp {
    background-color: #0d1117;
    color: #e6edf3;
}

/* Header/Hero Container */
.hero-header {
    background: linear-gradient(135deg, #161b22 0%, #1f2937 100%);
    border: 1px solid #30363d;
    border-radius: 14px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1.2rem;
}

.hero-logo {
    background: linear-gradient(135deg, #2563eb, #3b82f6);
    color: white;
    font-weight: 800;
    font-size: 1.6rem;
    width: 52px;
    height: 52px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35);
}

.hero-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #f0f6fc;
    margin: 0;
}

.hero-subtitle {
    font-size: 0.92rem;
    color: #8b949e;
    margin: 0.2rem 0 0 0;
}

/* Source Pills */
.source-pill {
    display: inline-flex;
    align-items: center;
    background-color: #21262d;
    border: 1px solid #30363d;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.78rem;
    color: #58a6ff;
    margin-right: 6px;
    margin-top: 6px;
}

.badge-oos {
    display: inline-flex;
    align-items: center;
    background-color: rgba(248, 81, 73, 0.15);
    border: 1px solid #f85149;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.78rem;
    color: #f85149;
    margin-top: 6px;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏢 Zyro Dynamics")
    st.caption("Internal HR Intelligence Portal")
    st.markdown("---")

    default_key = os.getenv("GROQ_API_KEY", "")
    api_key_input = st.text_input(
        "🔑 Groq API Key",
        value=default_key,
        type="password",
        help="Required for LLM answer generation. Model: LLaMA-3.3-70B"
    )

    if api_key_input:
        os.environ["GROQ_API_KEY"] = api_key_input
        st.success("API Key configured", icon="✅")
    else:
        st.info("No API Key entered. Running in local retrieval / fallback mode.", icon="ℹ️")

    st.markdown("---")
    st.markdown("#### ⚡ Quick Topics")
    quick_queries = [
        ("🏖️ Earned Leave Accrual", "At what rate does Earned Leave accrue per month at Acrux Dynamics?"),
        ("💰 L4 CTC & Bonus", "What is the CTC range and bonus target for an L4 Senior grade employee?"),
        ("🏠 WFH Policy Rules", "Who is eligible to work from home and what are the WFH arrangements?"),
        ("🏥 Health Insurance", "What health insurance coverage is provided to employees?"),
        ("📈 Performance & PIP", "When is an employee placed on a PIP and what is the duration?"),
    ]

    for label, query in quick_queries:
        if st.button(label, use_container_width=True):
            st.session_state.pending_query = query

    st.markdown("---")
    show_chunks = st.checkbox("🔍 Debug: Inspect Retrieved Chunks", value=False)

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("<small style='color:#8b949e;'>11 HR Policy Documents Loaded<br>FAISS CPU + MiniLM Embeddings</small>", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-logo">ZD</div>
    <div>
        <p class="hero-title">HR Help Desk Intelligence</p>
        <p class="hero-subtitle">Official policy assistant for Zyro Dynamics Pvt. Ltd. (Acrux Dynamics)</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Pipeline Caching ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading HR Vector Store & Retriever...")
def load_cached_retriever():
    vectorstore = build_or_load_vectorstore()
    retriever = get_retriever(vectorstore)
    return retriever


retriever = load_cached_retriever()

# ── Session State ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 Hello! I am your HR Help Desk Assistant for **Zyro Dynamics Pvt. Ltd.** "
                "You can ask me questions about internal policies, leave entitlements, health insurance, "
                "salary schedules, performance appraisals, or work-from-home guidelines. How can I help you today?"
            ),
            "sources": []
        }
    ]

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# Render Chat History
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🏢"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("sources"):
            sources_html = "".join([f"<span class='source-pill'>📄 {s}</span>" for s in msg["sources"]])
            st.markdown(f"<div style='margin-top:8px;'>{sources_html}</div>", unsafe_allow_html=True)
        if msg.get("blocked"):
            st.markdown("<span class='badge-oos'>🚫 Out of HR Policy Scope</span>", unsafe_allow_html=True)

# User Input
chat_input = st.chat_input("Ask a question about Zyro Dynamics HR policies...")
query_to_process = chat_input or st.session_state.pending_query
st.session_state.pending_query = None

if query_to_process:
    # Append User Message
    st.session_state.messages.append({"role": "user", "content": query_to_process})
    with st.chat_message("user", avatar="👤"):
        st.markdown(query_to_process)

    # Process Bot Response
    with st.chat_message("assistant", avatar="🏢"):
        with st.spinner("Searching verified HR policy documents..."):
            llm = get_llm(api_key=api_key_input)
            result = ask_bot(query_to_process, retriever=retriever, llm=llm)

            answer = result.get("answer", "")
            sources = result.get("sources", [])
            blocked = result.get("blocked", False)

            st.markdown(answer)

            if sources:
                sources_html = "".join([f"<span class='source-pill'>📄 {s}</span>" for s in sources])
                st.markdown(f"<div style='margin-top:8px;'>{sources_html}</div>", unsafe_allow_html=True)

            if blocked:
                st.markdown("<span class='badge-oos'>🚫 Out of HR Policy Scope</span>", unsafe_allow_html=True)

            if show_chunks and sources:
                with st.expander("🔍 Retrieved Context Chunks"):
                    chunks = retriever.invoke(query_to_process)
                    for i, c in enumerate(chunks, 1):
                        st.markdown(f"**Chunk {i} — {c.metadata.get('source')} (Page {c.metadata.get('page')})**")
                        st.text(c.page_content.strip())
                        st.divider()

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "blocked": blocked
            })
