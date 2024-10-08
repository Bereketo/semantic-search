# Movie Search with FAISS and Streamlit

Welcome to my project! This is a simple tool I built to search through a movie dataset using FAISS for fast similarity search, all wrapped up in a neat Streamlit app.

## What's Inside?

- **Semantic Search**: Type in a movie title, genre, or anything else, and find similar movies.
- **Speedy Results**: Thanks to FAISS, the search is super fast—even with large datasets.
- **Easy Interface**: The Streamlit UI is straightforward and user-friendly.

## What You'll Need

- Python 3.7+
- FAISS (for search)
- Streamlit (for the web app)
- Sentence Transformers (for generating embeddings)
- Pandas and NumPy (for data handling)

## How to Get Started

1. **Clone the Repo**  
   First, grab the code from GitHub:
   ```bash
   git clone https://github.com/bereketo/semantic-search.git
   cd semantic-search
   ```
2. **Set up your Environment**
   I recommend using a virtual environment
   ```bash
   python3 -m venv venv
   source venv/bin/activate 
   ```
3. **Install the necessary requirements**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**
   ```bash
   streamlit  app.py  or python -m streamlit run app.py
   ```
## IVF Integration
In this project, I have integrated IVG (Inverted File System with Graph) to significantly enhance the speed and performance of the nearest neighbor similarity search. IVG allows for faster, more scalable searches by efficiently handling large datasets and complex queries, solving the slowness that can occur with traditional methods.
## Screenshot

![Image description](images/movie-1.png)
![Image description](images/movie-2.png)
![Image description](images/movie-3.png)


