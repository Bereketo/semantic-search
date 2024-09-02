from sentence_transformers import SentenceTransformer


def generate_embeddings(df, model_name="all-mpnet-base-v2"):
    """Generate embeddings"""
    model = SentenceTransformer(model_name)
    df["Overview_Embeddings"] = df["Overview"].apply(lambda x: model.encode(x))
    return df, model
