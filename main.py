from data_loader import load_data, preprocess_data
from embeddings_generator import generate_embeddings
from faiss_index import build_faiss_index
from search import search_query


data_path = "./data/imdb_top_1000.csv"


def main():
    df = load_data(data_path)
    df_selected = preprocess_data(df)

    df_selected, model = generate_embeddings(df_selected)
    index = build_faiss_index(df_selected["Overview_Embeddings"])

    query = "A love movie"
    search_results = search_query(query, model, index, df_selected)

    print(search_results[["Series_Title", "Overview"]])


if __name__ == "__main__":
    main()
