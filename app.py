import streamlit as st
from data_loader import load_data, preprocess_data
from embeddings_generator import generate_embeddings
from faiss_index import build_faiss_index
from search import search_query


data_path = "./data/imdb_top_1000.csv"


@st.cache_data
def load_and_prepare_data():
    df = load_data(data_path)
    df_selected = preprocess_data(df)
    df_selected, model = generate_embeddings(df_selected)
    index = build_faiss_index(df_selected["Overview_Embeddings"])
    return df_selected, model, index


df_selected, model, index = load_and_prepare_data()

st.title("Movie Search App")

st.write("Enter a movie description or plot to find similar movies:")

query = st.text_input("Search query", "")

if query:
    st.write("Searching for movies similar to your query...")
    results = search_query(query, model, index, df_selected)

    st.write(f"Top {len(results)} results:")

    for idx, row in results.iterrows():
        st.subheader(row["Series_Title"])
        st.write(f"**Genre:** {row['Genre']}")
        st.write(f"**Director:** {row['Director']}")
        st.write(f"**Overview:** {row['Overview']}")
        st.write(
            f"**Stars:** {row['Star1']}, {row['Star2']}, {row['Star3']}, {row['Star4']}"
        )
        st.write("---")

if st.checkbox("Show dataset"):
    st.write(df_selected.head(10))
