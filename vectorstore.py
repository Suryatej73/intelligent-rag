import os
from pathlib import Path
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

import config
from ingest import load_documents, chunk_documents
from embeddings import get_embeddings_model

def build_or_load_vectorstore(
    force_rebuild: bool = False,
    index_path: Path = config.FAISS_INDEX_PATH
) -> FAISS:
    """
    Build a FAISS vectorstore from document chunks or load from local storage if available.
    """
    embeddings = get_embeddings_model()
    index_file = index_path / "index.faiss"

    if not force_rebuild and index_file.exists():
        print("\n" + "=" * 60)
        print("STAGE 5: LOADING EXISTING FAISS VECTORSTORE")
        print("=" * 60)
        print(f"Loading vectorstore from: {index_path}")
        try:
            vectorstore = FAISS.load_local(
                folder_path=str(index_path),
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
            print("FAISS vectorstore loaded successfully from disk.")
            return vectorstore
        except Exception as e:
            print(f"Failed to load cached index: {e}. Rebuilding from source documents...")

    print("\n" + "=" * 60)
    print("STAGE 5: BUILDING NEW FAISS VECTORSTORE")
    print("=" * 60)
    documents = load_documents(config.CORPUS_DIR)
    chunks = chunk_documents(documents)

    print(f"Generating embeddings for {len(chunks)} chunks and creating FAISS index...")
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)

    # Save to disk
    index_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_path))
    print(f"FAISS index saved successfully to: {index_path}")

    return vectorstore


def get_retriever(
    vectorstore: Optional[FAISS] = None,
    search_type: str = config.RETRIEVER_SEARCH_TYPE,
    k: int = config.RETRIEVER_K,
    fetch_k: int = config.RETRIEVER_FETCH_K,
    lambda_mult: float = config.RETRIEVER_LAMBDA_MULT
) -> VectorStoreRetriever:
    """
    Configure and return a Maximum Marginal Relevance (MMR) retriever.
    """
    if vectorstore is None:
        vectorstore = build_or_load_vectorstore()

    print("\n" + "=" * 60)
    print("STAGE 6: CONFIGURING MMR RETRIEVER")
    print("=" * 60)
    print(f"Search Type : {search_type}")
    print(f"k (top-k)   : {k}")
    print(f"fetch_k     : {fetch_k}")
    print(f"lambda_mult : {lambda_mult}")

    if search_type == "mmr":
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": fetch_k,
                "lambda_mult": lambda_mult
            }
        )
    else:
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )

    return retriever


if __name__ == "__main__":
    vs = build_or_load_vectorstore(force_rebuild=True)
    retriever = get_retriever(vs)

    test_query = "At what rate does Earned Leave accrue per month at Acrux Dynamics?"
    print(f"\nTesting retrieval for: '{test_query}'")
    results = retriever.invoke(test_query)
    print(f"Retrieved {len(results)} chunks:")
    for idx, doc in enumerate(results, 1):
        src = doc.metadata.get("source", "Unknown")
        pg = doc.metadata.get("page", "?")
        print(f"[{idx}] Source: {src} (Page {pg})")
        print(f"    Preview: {doc.page_content.strip()[:160]}...\n")

    assert len(results) > 0, "Retriever returned no chunks!"
    print("Vectorstore & Retriever verification PASSED.")
