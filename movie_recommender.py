!pip install streamlit

#streamlit 
# Import necessary libraries
import streamlit as st               # Streamlit for building the web app
import pandas as pd                # Pandas for data manipulation
import ast                         # For safely evaluating genre strings in JSON format
from sklearn.metrics.pairwise import cosine_similarity  # For calculating similarity

# Function to convert genre string to a list of genre names
def extract_genres(genre_str):
    try:
        genres = ast.literal_eval(genre_str)
        if isinstance(genres, list):
              return[genre.get("name", "") for genre in genres]
    except (ValueError, SyntaxError):
        pass
    return[]

# Load and preprocess the dataset
df = pd.read_csv('tmdb_5000_movies.csv')

# Create a new column with the extracted genres
df['genre_list'] = df['genres'].apply(extract_genres)

# One-hot encode the genre list into dummy variables
genre_dummies = df['genre_list'].apply(lambda x: pd.Series(1, index=x)).fillna(0)

# Merge the dummy variables into the original DataFrame
df_with_genres = pd.concat([df, genre_dummies], axis=1)

# Create a genre matrix and compute cosine similarity
genre_matrix = df_with_genres[genre_dummies.columns]
similarity = cosine_similarity(genre_matrix)

# Movie recommendation function
def recommend_movies(movie_title, top_n=5):
    if movie_title not in df_with_genres['title'].values:
        return []
    idx = df_with_genres[df_with_genres['title'] == movie_title].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in scores]
    return df_with_genres.iloc[movie_indices]

# Streamlit UI setup
st.set_page_config(page_title="Movie Recommender", layout="wide")

# App Title
st.title("Movie Recommendation App")

# ℹ Sidebar Info
with st.sidebar:
    st.header("ℹ About")
    st.markdown("""
    This app recommends movies based on genre similarity using cosine similarity.

    - Dataset: TMDB 5000 Movies
    - Method: Genre-based similarity
    """)

# User selects a movie
selected_movie = st.selectbox("Choose a movie:", df_with_genres['title'].sort_values().unique())

# Recommend button
if st.button("Recommend"):
    recommendations = recommend_movies(selected_movie)

    if len(recommendations) > 0:
        st.subheader(f"Top 5 movies similar to: **{selected_movie}**")
        for _, row in recommendations.iterrows():
            st.markdown(f"**{row['title']}**")
            st.markdown(f"Release Date: `{row['release_date']}`")
            st.markdown(f"Genres: `{', '.join(row['genre_list'])}`")
            st.markdown("---")
    else:
        st.warning("⚠️ Movie not found or not enough data to recommend.")
