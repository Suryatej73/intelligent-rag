import os
import re
from pathlib import Path
from dotenv import load_dotenv

# Load local .env if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# Check if running in Kaggle environment
IS_KAGGLE = os.path.exists("/kaggle/input")

if IS_KAGGLE:
    KAGGLE_CORPUS_CANDIDATES = [
        Path("/kaggle/input/competitions/gen-ai-project-intelligent-rag/zyro-dynamics-hr-corpus"),
        Path("/kaggle/input/gen-ai-project-intelligent-rag/zyro-dynamics-hr-corpus"),
        Path("/kaggle/input/competitions/gen-ai-project-intelligent-rag"),
        Path("/kaggle/input/zyro-dynamics-hr-corpus"),
    ]
    CORPUS_DIR = next((p for p in KAGGLE_CORPUS_CANDIDATES if p.exists()), KAGGLE_CORPUS_CANDIDATES[0])
    DATA_DIR = CORPUS_DIR if CORPUS_DIR.exists() else Path("/kaggle/input")
    VECTOR_DB_DIR = Path("/kaggle/working/vector_db")
    OUTPUT_DIR = Path("/kaggle/working")
    TEST_CSV_PATH = DATA_DIR / "test.csv"
else:
    CORPUS_DIR = BASE_DIR / "data"
    DATA_DIR = BASE_DIR / "data"
    VECTOR_DB_DIR = BASE_DIR / "vector_db"
    OUTPUT_DIR = BASE_DIR / "outputs"
    TEST_CSV_PATH = DATA_DIR / "test.csv"

# Ensure output directories exist
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 11 Official HR Documents
EXPECTED_DOCUMENTS = [
    "00_Company_Profile.pdf",
    "01_Employee_Handbook.pdf",
    "02_Leave_Policy.pdf",
    "03_Work_From_Home_Policy.pdf",
    "04_Code_of_Conduct.pdf",
    "05_Performance_Review_Policy.pdf",
    "06_Compensation_and_Benefits_Policy.pdf",
    "07_IT_and_Data_Security_Policy.pdf",
    "08_Prevention_of_Sexual_Harassment_Policy.pdf",
    "09_Onboarding_and_Separation_Policy.pdf",
    "10_Travel_and_Expense_Policy.pdf",
]

DOC_LABELS = {
    "00_Company_Profile.pdf": "Company Profile",
    "01_Employee_Handbook.pdf": "Employee Handbook",
    "02_Leave_Policy.pdf": "Leave Policy",
    "03_Work_From_Home_Policy.pdf": "Work From Home Policy",
    "04_Code_of_Conduct.pdf": "Code of Conduct",
    "05_Performance_Review_Policy.pdf": "Performance Review Policy",
    "06_Compensation_and_Benefits_Policy.pdf": "Compensation & Benefits Policy",
    "07_IT_and_Data_Security_Policy.pdf": "IT & Data Security Policy",
    "08_Prevention_of_Sexual_Harassment_Policy.pdf": "POSH Policy",
    "09_Onboarding_and_Separation_Policy.pdf": "Onboarding & Separation Policy",
    "10_Travel_and_Expense_Policy.pdf": "Travel & Expense Policy",
}

# Embedding Model Configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"
EMBEDDING_NORMALIZE = True

# Text Chunking Configuration
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
SEPARATORS = ["\n\n\n", "\n\n", "\n", ". ", ", ", " ", ""]

# Vector Store Path
FAISS_INDEX_PATH = VECTOR_DB_DIR / "faiss_index"

# MMR Retriever Configuration
RETRIEVER_SEARCH_TYPE = "mmr"
RETRIEVER_K = 5
RETRIEVER_FETCH_K = 15
RETRIEVER_LAMBDA_MULT = 0.7

# LLM Configuration
PREFERRED_LLM_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
FALLBACK_LLM_MODEL = "llama-3.1-8b-instant"
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 512

# Competition Secret Key for Question Decryption and Submission Verification
SUBMISSION_SECRET = b"6Q_EBPtBG-60URcrF6jxNTJSRjy-CtZbJlvp_xf0c_M="

# Company Entities
COMPANY_NAME = "Zyro Dynamics Pvt. Ltd."
COMPANY_ALIAS = "Acrux Dynamics"

# Standard Guardrail Refusal Message
REFUSAL_MESSAGE = (
    "I'm sorry, I can assist with HR-related queries concerning Zyro Dynamics Pvt. Ltd. "
    "(Acrux Dynamics) policies. Please ask me a question related to company HR policies."
)

# Missing Information Fallback Message
MISSING_INFO_MESSAGE = (
    "I'm sorry, but the provided HR policy documents do not contain sufficient information "
    "to answer this question."
)

def get_groq_api_key():
    """Retrieve Groq API key from environment, .env, or Kaggle Secrets."""
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key and IS_KAGGLE:
        try:
            from kaggle_secrets import UserSecretsClient
            secrets = UserSecretsClient()
            key = secrets.get_secret("GROQ_API_KEY")
        except Exception:
            pass
    return key
