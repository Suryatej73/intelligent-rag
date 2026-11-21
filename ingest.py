import os
import glob
from pathlib import Path
from typing import List, Dict, Tuple
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

def load_documents(corpus_dir: Path = config.CORPUS_DIR) -> List[Document]:
    """
    Load all 11 HR policy PDF documents from the specified directory.
    Validates file existence, readable pages, and attaches metadata.
    """
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found at: {corpus_dir}")

    pdf_files = sorted(glob.glob(str(corpus_dir / "*.pdf")))
    if not pdf_files:
        raise ValueError(f"No PDF files found in {corpus_dir}")

    print("\n" + "=" * 60)
    print("STAGE 2: PDF DOCUMENT LOADING & VALIDATION")
    print("=" * 60)
    print(f"Reading from: {corpus_dir}")
    print(f"Found {len(pdf_files)} PDF file(s). Expected: {len(config.EXPECTED_DOCUMENTS)}")

    # Check for missing files
    found_filenames = {os.path.basename(p) for p in pdf_files}
    missing_expected = [f for f in config.EXPECTED_DOCUMENTS if f not in found_filenames]
    if missing_expected:
        print(f"⚠️ Warning: Missing {len(missing_expected)} expected document(s): {missing_expected}")

    all_pages: List[Document] = []
    loading_summary: List[Dict] = []
    seen_hashes = set()

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        loader = PyPDFLoader(pdf_path)
        try:
            pages = loader.load()
        except Exception as e:
            print(f"❌ Failed to load {filename}: {e}")
            continue

        valid_pages = []
        empty_pages_count = 0

        for page_idx, page in enumerate(pages):
            content = page.page_content.strip()
            if not content:
                empty_pages_count += 1
                continue

            # Standardize metadata
            page.metadata["source"] = filename
            page.metadata["page"] = page_idx + 1  # 1-indexed page number
            page.metadata["doc_title"] = config.DOC_LABELS.get(filename, filename.replace(".pdf", ""))
            valid_pages.append(page)

        all_pages.extend(valid_pages)
        loading_summary.append({
            "filename": filename,
            "title": config.DOC_LABELS.get(filename, filename),
            "total_pages": len(pages),
            "valid_pages": len(valid_pages),
            "empty_pages": empty_pages_count,
        })

    # Display summary
    print(f"\n{'Document Name':<42} | {'Pages':<6} | {'Status'}")
    print("-" * 65)
    for item in loading_summary:
        status = "OK" if item["empty_pages"] == 0 else f"{item['empty_pages']} empty page(s)"
        print(f"{item['filename']:<42} | {item['valid_pages']:<6} | {status}")
    print("-" * 65)
    print(f"Total Pages Extracted: {len(all_pages)}\n")

    return all_pages


def chunk_documents(
    documents: List[Document],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
    separators: List[str] = config.SEPARATORS
) -> List[Document]:
    """
    Chunk documents using RecursiveCharacterTextSplitter, preserving metadata.
    """
    print("=" * 60)
    print("STAGE 3: TEXT CHUNKING")
    print("=" * 60)
    print(f"Configuration: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        is_separator_regex=False
    )

    chunks = splitter.split_documents(documents)
    # Filter out trivial chunks
    chunks = [c for c in chunks if len(c.page_content.strip()) > 30]

    total_chunks = len(chunks)
    avg_chunk_size = sum(len(c.page_content) for c in chunks) / max(total_chunks, 1)

    print(f"Total Extracted Pages : {len(documents)}")
    print(f"Total Chunks Created  : {total_chunks}")
    print(f"Average Chunk Size    : {avg_chunk_size:.1f} characters")

    if chunks:
        sample = chunks[0]
        print("\n--- Example Chunk [Metadata & Preview] ---")
        print(f"Source Document : {sample.metadata.get('source')}")
        print(f"Page Number     : {sample.metadata.get('page')}")
        print(f"Content Length  : {len(sample.page_content)} characters")
        print(f"Preview         :\n{sample.page_content[:250]}...\n")

    return chunks


if __name__ == "__main__":
    docs = load_documents()
    chunks = chunk_documents(docs)
    print(f"Document ingestion test passed with {len(chunks)} chunks.")
