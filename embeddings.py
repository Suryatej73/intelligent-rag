from typing import Optional
from langchain_huggingface import HuggingFaceEmbeddings
import config

_EMBEDDINGS_INSTANCE: Optional[HuggingFaceEmbeddings] = None

def get_embeddings_model() -> HuggingFaceEmbeddings:
    """
    Initialize and return the cached HuggingFace sentence transformer embedding model.
    Model: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
    """
    global _EMBEDDINGS_INSTANCE
    if _EMBEDDINGS_INSTANCE is None:
        print("\n" + "=" * 60)
        print("STAGE 4: EMBEDDING MODEL INITIALIZATION")
        print("=" * 60)
        print(f"Model Name : {config.EMBEDDING_MODEL_NAME}")
        print(f"Device     : {config.EMBEDDING_DEVICE}")
        print(f"Normalized : {config.EMBEDDING_NORMALIZE}")

        _EMBEDDINGS_INSTANCE = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL_NAME,
            model_kwargs={"device": config.EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": config.EMBEDDING_NORMALIZE}
        )
        print("Embedding model loaded successfully.\n")

    return _EMBEDDINGS_INSTANCE


if __name__ == "__main__":
    embeddings = get_embeddings_model()
    test_text = "What is the leave policy at Zyro Dynamics?"
    vector = embeddings.embed_query(test_text)
    print(f"Test query embedded successfully.")
    print(f"Vector dimension: {len(vector)} (Expected: 384)")
    assert len(vector) == 384, f"Unexpected embedding dimension: {len(vector)}"
    print("Embedding verification PASSED.")
