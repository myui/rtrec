import os
import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta

# URLs for MovieLens 100K dataset
DATASET_URL = "http://files.grouplens.org/datasets/movielens/ml-100k/"
RATINGS_FILE = DATASET_URL + "u.data"
ITEMS_FILE = DATASET_URL + "u.item"

# Load ratings and item data
@st.cache_data
def load_movielens_100k():
    col_rat = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings_df = pd.read_csv(RATINGS_FILE, sep='\t', names=col_rat)

    col_items = [
        'movie_id', 'movie_title', 'release_date', 'video_release_date', 
        'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation', 
        'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 
        'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci_Fi', 
        'Thriller', 'War', 'Western'
    ]
    items_df = pd.read_csv(ITEMS_FILE, sep='|', names=col_items, encoding='latin-1')

    # Convert unix timestamps to datetime
    ratings_df['timestamp'] = pd.to_datetime(ratings_df['unix_timestamp'], unit='s')

    # Merge ratings and item data on movie_id
    merged_df = pd.merge(ratings_df, items_df[['movie_id', 'movie_title']], on='movie_id', how='left')

    return ratings_df, items_df, merged_df

@st.cache_data
def load_movie_posters():
    # see https://github.com/babu-thomas/movielens-posters
    poster_url = "https://raw.githubusercontent.com/babu-thomas/movielens-posters/refs/heads/master/movie_poster.csv"
    poster_df = pd.read_csv(poster_url, header=None, names=['movie_id', 'url'])
    
    # Create a dictionary to store movie_id as key and poster_url as value
    poster_dict = {}
    for _, row in poster_df.iterrows():
        poster_dict[row['movie_id']] = row['url']
    return poster_dict

# Recommend movies for a given user
def recommend(user_id, merged_df, feedback, top_n=5, cutoff_days=180):
    recent_date = datetime.now() - timedelta(days=cutoff_days)
    recent_data = merged_df[merged_df["timestamp"] >= recent_date]

    # Exclude movies the user has already seen
    seen_movies = recent_data[recent_data["user_id"] == user_id]["movie_id"].unique()
    recommendations = merged_df[~merged_df["movie_id"].isin(seen_movies)].drop_duplicates("movie_id")

    # Randomly select recommendations for simplicity
    return recommendations.sample(n=top_n)

# Streamlit UI
st.title("MovieLens 100K Recommender System")
st.sidebar.header("User Input")

# Load data
ratings_df, items_df, merged_df = load_movielens_100k()

# Load movie poster data from the CSV file
poster_dict = load_movie_posters()

# User input and recommendation configuration
user_id = st.sidebar.number_input("Enter User ID", min_value=1, value=1)
top_n = st.sidebar.slider("Number of Recommendations", min_value=1, max_value=10, value=5)
cutoff_days = st.sidebar.slider("Recent Interactions (days)", min_value=30, max_value=365, value=180)

if "feedback" not in st.session_state:
    st.session_state.feedback = {"liked": [], "disliked": []}

if st.button("Get Recommendations"):
    recommendations = recommend(user_id, merged_df, st.session_state.feedback, top_n, cutoff_days)

    st.write(f"Top {top_n} recommendations for User {user_id}:")

    for _, row in recommendations.iterrows():
        title = row["movie_title"]
        movie_id = row["movie_id"]

        st.write(f"**{title}**")
        
        poster_url = poster_dict.get(movie_id)
        if poster_url:
            st.image(poster_url, width=100)
        else:
            st.write("No image available.")

        liked = st.checkbox(f"👍 Like {title}", key=f"like_{row['movie_id']}")
        disliked = st.checkbox(f"👎 Dislike {title}", key=f"dislike_{row['movie_id']}")

        if liked:
            st.session_state.feedback["liked"].append(title)
        elif disliked:
            st.session_state.feedback["disliked"].append(title)

    if st.session_state.feedback:
        st.write("Updating recommendations based on your feedback...")
        new_recommendations = recommend(user_id, merged_df, st.session_state.feedback, top_n, cutoff_days)
        st.write("Updated Recommendations:")
        for _, row in new_recommendations.iterrows():
            st.write(f"- {row['movie_title']}")

# Display raw data (optional)
if st.sidebar.checkbox("Show Raw Data"):
    st.write("Ratings Data", ratings_df.head())
    st.write("Items Data", items_df.head())
