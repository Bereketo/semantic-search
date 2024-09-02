import faiss
import numpy as np


def build_faiss_index(embeddings):
    """Build and return faiss index"""
    dimensions = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimensions)
    embeddings_array = np.array(embeddings.tolist())
    index.add(embeddings_array)
    return index
