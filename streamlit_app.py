import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score
)
from sklearn.metrics.pairwise import cosine_similarity

# Load data
df = pd.read_csv("tmdb_5000_movies.csv")

# Extract genres from string to list
def extract_genres(genre_str):
    try:
        genres = ast.literal_eval(genre_str)
        return [genre.get("name", "") for genre in genres]
    except:
        return []

df['genre_list'] = df['genres'].apply(extract_genres)

# One-hot encode genres
genre_dummies = df['genre_list'].apply(lambda x: pd.Series(1, index=x)).fillna(0)
df_with_genres = pd.concat([df, genre_dummies], axis=1)

# Feature engineering
df_with_genres['release_year'] = pd.to_datetime(df_with_genres['release_date'], errors='coerce').dt.year
df_with_genres['weighted_score'] = df_with_genres['vote_average'] * df_with_genres['vote_count']

# Select features
selected_features_df = df_with_genres[[
    'id', 'title', 'genres', 'keywords', 'cast', 'crew', 'release_date',
    'runtime', 'vote_average', 'vote_count', 'weighted_score', 'release_year'
]]

# Binning vote_average
bins = [0, 5, 6, 7, 8, 10]
labels = [1, 2, 3, 4, 5]
selected_features_df = selected_features_df.copy()
selected_features_df['vote_average_binned'] = pd.cut(
    selected_features_df['vote_average'], bins=bins, labels=labels
)

# Clean NaNs
selected_features_df_clean = selected_features_df.dropna(subset=['vote_average_binned'])

# Prepare X and y
X = selected_features_df_clean.drop(columns=[
    'vote_average', 'vote_average_binned',
    'release_date', 'title', 'genres', 'keywords', 'cast', 'crew'
])
y = selected_features_df_clean['vote_average_binned']

# One-hot encode if needed (no categoricals remain now)
X.fillna(0, inplace=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
y_pred_binned = pd.cut(y_pred, bins=bins, labels=labels)

accuracy = accuracy_score(y_test, y_pred_binned)
precision = precision_score(y_test, y_pred_binned, average='weighted')
recall = recall_score(y_test, y_pred_binned, average='weighted')
f1 = f1_score(y_test, y_pred_binned, average='weighted')

# Recommender system
genre_columns = genre_dummies.columns
genre_matrix = df_with_genres[genre_columns]
similarity = cosine_similarity(genre_matrix)

def recommend_movies(movie_title, df, similarity_matrix, top_n=5):
    if movie_title not in df['title'].values:
        return f"Movie '{movie_title}' not found."

    idx = df[df['title'] == movie_title].index[0]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in scores]
    return df['title'].iloc[movie_indices]

# Streamlit UI
st.set_page_config(page_title=" Movie Recommender", layout="wide")

st.title(" Movie Recommendation System")
st.markdown("Recommend movies based on genre similarity using TMDB data.")

with st.sidebar:
    st.header("ℹAbout")
    st.markdown("""
    - Dataset: TMDB 5000 Movies
    - Model: Content-Based Filtering
    - Techniques: Cosine Similarity, Random Forest
    """)

st.subheader(" Model Evaluation")
st.write(f"**Accuracy:** {accuracy:.2f}")
st.write(f"**Precision:** {precision:.2f}")
st.write(f"**Recall:** {recall:.2f}")
st.write(f"**F1 Score:** {f1:.2f}")
st.write(f"**Mean Squared Error:** {mse:.2f}")
st.write(f"**R² Score:** {r2:.2f}")

st.subheader(" Get Movie Recommendations")
selected_movie = st.selectbox("Choose a movie:", df_with_genres['title'].dropna().unique())

if st.button("Recommend"):
    results = recommend_movies(selected_movie, df_with_genres, similarity)
    if isinstance(results, str):
        st.warning(results)
    else:
        st.write(f"Movies similar to **{selected_movie}**:")
        for title in results:
            st.markdown(f"- {title}")