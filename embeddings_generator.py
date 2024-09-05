from sentence_transformers import SentenceTransformer
import numpy as np
from datasets import Dataset
from faiss_index import build_faiss_index
import faiss


def generate_embeddings(df, model_name="all-mpnet-base-v2"):
    """Generate embeddings using SentenceTransformer model."""
    model = SentenceTransformer(model_name)
    df["Overview_Embeddings"] = df["Overview"].apply(
        lambda x: model.encode(x).astype(np.float32)
    )
    return df, model


"""
def prepare_rag_data(df, dataset_path, index_path):
    
    # Ensure the embeddings are in the correct format
    df["Overview_Embeddings"] = df["Overview_Embeddings"].apply(
        lambda x: np.array(x, dtype=np.float32)
    )

    # Create the required columns for RAG
    df_rag = df.rename(columns={"Series_Title": "title", "Overview": "text"})
    df_rag["embeddings"] = df["Overview_Embeddings"]

    # Convert DataFrame to a Hugging Face Dataset
    rag_dataset = Dataset.from_pandas(df_rag[["title", "text", "embeddings"]])

    # Save dataset to disk
    rag_dataset.save_to_disk(dataset_path)

    # Create and save FAISS index
    rag_dataset.add_faiss_index(column="embeddings", index_name="embeddings")
    rag_dataset.get_index("embeddings").save(index_path)

    return rag_dataset
    """
