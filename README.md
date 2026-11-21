# Zyro Dynamics HR Help Desk — Intelligent RAG System
**Kaggle Competition:** Gen AI Project: Intelligent RAG – NxtWave Academy  
**Organization:** Zyro Dynamics Pvt. Ltd. (Acrux Dynamics)

---

## 📌 Executive Overview
This project delivers a complete enterprise-grade **Retrieval-Augmented Generation (RAG)** chatbot designed to answer employee questions strictly based on **11 internal HR policy documents**.

The solution implements high-precision document chunking, semantic embeddings with HuggingFace MiniLM, FAISS vector indexing, Maximum Marginal Relevance (MMR) retrieval, LLaMA-3.3-70B via Groq with LangChain LCEL, input scope guardrails, automated evaluation, and compliant Kaggle submission generation.

---

## 🏗️ Architecture & Tech Stack

```
                                  [ Employee Query ]
                                          │
                                   [ Guardrails ]
                             ┌────────────┴────────────┐
                      [ In-Scope ]              [ Out-of-Scope ]
                             │                         │
                   [ FAISS Vector Store ]         [ Polite Refusal ]
                             │
                      [ MMR Retrieval ]
                    (k=5, fetch_k=15)
                             │
                      [ Context Chunks ]
                             │
                 [ LangChain LCEL + Prompt ]
                             │
                 [ Groq: LLaMA-3.3-70B LLM ]
                             │
                    [ Grounded Answer ]
                   + [ Source Citations ]
```

- **Framework:** LangChain & LangChain Expression Language (LCEL)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, normalized)
- **Vector Store:** FAISS CPU
- **Retrieval Strategy:** Maximum Marginal Relevance (MMR: `k=5`, `fetch_k=15`, `lambda_mult=0.7`)
- **LLM Provider:** Groq API (`llama-3.3-70b-versatile` / `llama-3.1-8b-instant`, temperature=0.0)
- **Frontend UI:** Streamlit (`app.py`)
- **Security:** Guardrails preventing hallucinations, handling missing context, refusing out-of-scope inquiries

---

## 📂 Project Structure

```
intelligent-rag/
├── app.py                   # Streamlit HR Help Desk web application
├── config.py                # Centralized environment, paths & model configurations
├── ingest.py                # PDF document loader & text chunker
├── embeddings.py            # Sentence-transformer embedding model wrapper
├── vectorstore.py           # FAISS vector store builder and MMR retriever
├── rag_chain.py             # LCEL RAG chain, guardrails & ask_bot() function
├── evaluation.py            # Evaluation benchmark & audit report generator
├── generate_submission.py   # Encrypted & plain Kaggle submission generator
├── requirements.txt         # Project dependencies
├── .env.example             # Template for API keys
├── .gitignore               # Git rules excluding secrets, venvs, and indices
├── Starter Notebook.ipynb   # Verified Kaggle Starter Notebook
├── data/                    # 11 Official HR PDF documents & test datasets
│   ├── 00_Company_Profile.pdf
│   ├── 01_Employee_Handbook.pdf
│   ├── 02_Leave_Policy.pdf
│   ├── 03_Work_From_Home_Policy.pdf
│   ├── 04_Code_of_Conduct.pdf
│   ├── 05_Performance_Review_Policy.pdf
│   ├── 06_Compensation_and_Benefits_Policy.pdf
│   ├── 07_IT_and_Data_Security_Policy.pdf
│   ├── 08_Prevention_of_Sexual_Harassment_Policy.pdf
│   ├── 09_Onboarding_and_Separation_Policy.pdf
│   ├── 10_Travel_and_Expense_Policy.pdf
│   ├── test.csv
│   └── sample_submission.csv
├── vector_db/               # Saved FAISS index files
│   └── faiss_index/
└── outputs/                 # Output artifacts
    ├── submission.csv       # Kaggle-ready encrypted 5-column submission
    ├── submission_plain.csv # Human-readable 2-column answers
    └── evaluation_report.csv# Test benchmark report
```

---

## 🚀 Quickstart & Local Setup

### 1. Create Virtual Environment & Install Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
```
Inside `.env`:
```ini
GROQ_API_KEY=gsk_your_actual_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
```

### 3. Build Vector Store & Verify Pipeline
```bash
python ingest.py
python embeddings.py
python vectorstore.py
python rag_chain.py
```

### 4. Run Benchmark Evaluation
```bash
python evaluation.py
```

### 5. Generate Submission Files
```bash
python generate_submission.py
```
This generates:
- `outputs/submission.csv` (Competition format: `question_id`, `question_enc`, `answer_enc`, `streamlit_link`, `langsmith_link`)
- `outputs/submission_plain.csv` (Inspection format: `question_id`, `answer`)

### 6. Launch the Streamlit App
```bash
streamlit run app.py
```

---

## 🏆 Kaggle Execution Guide

When running on Kaggle Notebooks:

1. **Notebook Settings:**
   - **Accelerator:** CPU (or GPU P100 if preferred)
   - **Internet:** **Must be toggled ON** (Settings panel -> Internet ON) to connect to Groq API and download the MiniLM weights.
2. **Kaggle Secrets:**
   - In Kaggle Notebook menu: `Add-ons` -> `Secrets`
   - Add label: `GROQ_API_KEY` with your Groq API key value.
   - (Optional) Add label: `LANGCHAIN_API_KEY` for LangSmith tracing.
3. **Execution Steps in Notebook:**
   - Run **Cell 1** to install dependencies (`pip install ...`).
   - Run **Cell 2 & 3** for configuration and imports.
   - Run **Cell 4** to load Kaggle Secrets.
   - Run **Cells 5 to 11** to load corpus PDFs, build vector index, and initialize `ask_bot()`.
   - Run **Cell 15** to load `eval_questions`.
   - Run **Cell 16** to run `ask_bot()` on all questions and generate `/kaggle/working/submission.csv`.
   - Run **Cell 17** to verify `submission.csv` format and link validations.
4. **Submit to Competition:**
   - Submit `/kaggle/working/submission.csv`.
