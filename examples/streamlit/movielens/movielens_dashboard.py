import datetime
import os
import time
from typing import Dict, Optional, Tuple

import streamlit as st
import pandas as pd
import requests
import re
import urllib.parse

from rtrec.models import SLIM
from rtrec.recommender import Recommender

OMDB_API_KEY = os.getenv("OMDB_API_KEY")

# URLs for MovieLens 100K dataset
DATASET_URL = "http://files.grouplens.org/datasets/movielens/ml-100k/"
RATINGS_FILE = DATASET_URL + "u.data"
ITEMS_FILE = DATASET_URL + "u.item"
USER_FILE = DATASET_URL + "u.user"

# Caching functions
@st.cache_data
def load_movielens_100k() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the MovieLens 100K dataset and merge ratings with movie details."""
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

    ratings_df['timestamp'] = pd.to_datetime(ratings_df['unix_timestamp'], unit='s')
    merged_df = pd.merge(ratings_df, items_df[['movie_id', 'movie_title']], on='movie_id', how='left')

    return ratings_df, items_df, merged_df

@st.cache_data
def load_user_data() -> pd.DataFrame:
    """Load the user demographic data from the MovieLens 100K dataset."""
    col_user = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
    user_df = pd.read_csv(USER_FILE, sep='|', names=col_user, encoding='latin-1')
    return user_df

@st.cache_data
def load_movie_posters() -> Dict[int, str]:
    """Load movie posters from an external source."""
    poster_url = "https://raw.githubusercontent.com/babu-thomas/movielens-posters/refs/heads/master/movie_poster.csv"
    poster_df = pd.read_csv(poster_url, header=None, names=['movie_id', 'url'])
    return dict(zip(poster_df['movie_id'], poster_df['url']))

def fetch_movie_details(title: str) -> Tuple[Optional[str], str]:
    """Fetch movie poster and plot from OMDB API."""
    def extract_title_year(movie_str):
        match = re.match(r'^(.*)\s\((\d{4})\)$', movie_str)
        if match:
            return match.group(1).strip(), match.group(2)
        return movie_str, None

    title, year = extract_title_year(title)
    query = urllib.parse.quote_plus(title)
    url = f"http://www.omdbapi.com/?t={query}&apikey={OMDB_API_KEY}"
    if year:
        url += f"&y={year}"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data.get("Response") == "True":
                plot = data.get("Plot")
                director = data.get("Director")
                genre = data.get("Genre")
                descrioption = f"{plot}\n\n**Director:** {director}\n**Genre:** {genre}"
                return data.get("Poster"), descrioption
            return None, data.get("Error", "No description available.")
    except requests.RequestException:
        return None, "Error fetching movie details."

    return None, "No description available."

def recommend_movies(user_id: int, df: pd.DataFrame, feedback: Dict[str, list], top_k: int = 5) -> pd.DataFrame:
    """Generate movie recommendations excluding already seen movies."""
    return recommender.recommend(user_id, top_k=top_k, filter_interacted=filter_watched)

@st.cache_resource
def recommender_init() -> Recommender:
    """Initialize the SLIM recommender model."""
    # Build Model
    model = SLIM(min_value=-5, max_value=10, decay_in_days=365)
    recommender = Recommender(model=model)
    recommender.bulk_fit(ratings_df.rename(columns={"user_id": "user", "movie_id": "item", "unix_timestamp": "tstamp"}), update_interaction=True, parallel=False)
    return recommender

@st.cache_data
def load_movie_titles() -> Dict[int, str]:
    """Load movie titles from the MovieLens 100K dataset."""
    return dict(zip(items_df['movie_id'], items_df['movie_title']))

def display_movie(title: str, movie_id: int, poster_dict: Dict[int, str]) -> None:
    """Display movie details including poster and plot with like/dislike feedback."""
    poster, plot = fetch_movie_details(title)
    if poster is None:
        poster = poster_dict.get(movie_id)

    st.write(f"**{title}**")
    col1, col2 = st.columns([1, 3])

    with col1:
        if poster is not None and poster != "N/A":
            st.image(poster, width=100)
        else:
            st.write("No image available.")

    with col2:
        st.write(plot)

    # Add movie to feedback only if it is newly liked/disliked
    liked_key = f"like_{movie_id}"
    disliked_key = f"dislike_{movie_id}"

    feedback_id = f"feedback_{movie_id}"
    feedback_options = ["👍 Like", "👎 Dislike"]
    feedback_unixtime = float(time.mktime(feedback_date.timetuple()))

    def handle_feedback():
        selected_index = feedback_options.index(st.session_state[feedback_id])
        if selected_index == 0:
            recommender.partial_fit([(user_id, movie_id, 5, feedback_unixtime)], update_interaction=False)
            st.session_state.feedback["liked"].append(title)
        elif selected_index == 1:
            st.session_state.feedback["disliked"].append(title)
            recommender.partial_fit([(user_id, movie_id, -5, feedback_unixtime)], update_interaction=False)
        st.session_state.feedback["date"].append(feedback_date)

    st.radio("Feedback", options=feedback_options, key=feedback_id, index=None, horizontal=True, on_change=handle_feedback)

# Streamlit UI
st.title("MovieLens 100K Recommender System")
st.sidebar.header("User Input")

# Load data
ratings_df, items_df, merged_df = load_movielens_100k()
user_df = load_user_data()
poster_dict = load_movie_posters()
movie_title_dict = load_movie_titles()

# Initialize recommender
recommender = recommender_init()

# User input
user_id = st.sidebar.number_input("Enter User ID", min_value=1, value=1)
if "last_user_id" not in st.session_state:
    st.session_state.last_user_id = user_id
# Initialize session state
if user_id != st.session_state.last_user_id:
    st.session_state.feedback = {"liked": [], "disliked": [], "date": []}
    st.session_state.last_user_id = user_id
elif "feedback" not in st.session_state:
    st.session_state.feedback = {"liked": [], "disliked": [], "date": []}
top_n = st.sidebar.slider("Number of Recommendations", min_value=1, max_value=10, value=5)
recent_feedback_n = st.sidebar.slider("Recent Feedback Items (N)", min_value=1, max_value=20, value=5)

feedback_date = st.sidebar.date_input("Feedback date", datetime.date(1998, 4, 23))

filter_watched = True
if not st.sidebar.checkbox("Filter watched movies", value=True):
    filter_watched = False

# Display raw data
if st.sidebar.checkbox("Show Raw Data"):
    st.write("Ratings Data", ratings_df.head())
    st.write("Items Data", items_df.head())

# Tabs for navigation
tab1, tab2, tab3 = st.tabs(["Recommendations", "Recently Feedbacked Movies", "Search Movies"])

with tab1:
    # Get recommendations
    recommendations = recommend_movies(user_id, merged_df, st.session_state.feedback, top_n)
    st.write(f"Top {top_n} recommendations for User {user_id}:")

    for movie_id in recommendations:
        display_movie(movie_title_dict[movie_id], movie_id, poster_dict)

with tab2:
    st.write("### User Information and Recently Feedbacked Movies")

    # Fetch user details
    user_info = user_df[user_df["user_id"] == user_id]
    if not user_info.empty:
        cols = st.columns(4)  # Create four columns for horizontal layout
        cols[0].write(f"**User ID:** {user_id}")
        cols[1].write(f"**Gender:** {user_info.iloc[0]['gender']}")
        cols[2].write(f"**Age:** {user_info.iloc[0]['age']}")
        cols[3].write(f"**Occupation:** {user_info.iloc[0]['occupation']}")
    else:
        st.write(f"User ID {user_id} not found in the dataset.")

    st.divider()

    # Get recently rated movies
    user_ratings = ratings_df[ratings_df["user_id"] == user_id]
    if not user_ratings.empty:
        user_ratings = user_ratings.sort_values(by="unix_timestamp", ascending=False)
        recent_ratings = user_ratings.merge(items_df[["movie_id", "movie_title"]], on="movie_id").head(recent_feedback_n)

        # Format data for display
        recent_ratings["date_rated"] = pd.to_datetime(recent_ratings["unix_timestamp"], unit="s").dt.date
        recent_ratings_table = recent_ratings[["movie_title", "rating", "date_rated"]].rename(
            columns={
                "movie_title": "Movie Title",
                "rating": "Rating",
                "date_rated": "Date Rated",
            }
        )

        st.write(f"#### Last {recent_feedback_n} Rated Movies")
        st.dataframe(recent_ratings_table, use_container_width=True)
    else:
        st.write("No ratings found for this user.")

    st.divider()

    liked_movies = st.session_state.feedback["liked"][-recent_feedback_n:]
    disliked_movies = st.session_state.feedback["disliked"][-recent_feedback_n:]
    feedback_dates = st.session_state.feedback["date"][-recent_feedback_n:]

    if liked_movies or disliked_movies:
        if liked_movies:
            st.write("#### Liked Movies")
            for movie, date in zip(liked_movies, feedback_dates):
                st.write(f"👍 {movie} at 🗓️ {date})")

        if disliked_movies:
            st.write("#### Disliked Movies")
            for movie, date in zip(disliked_movies, feedback_dates):
                st.write(f"👎 {movie} at 🗓️ {date}")
    else:
        st.write("No feedback given yet.")

with tab3:
    st.header("Search Movies by Title")
    search_query = st.text_input("Enter part of the movie title", "")

    if search_query:
        matching_movies = items_df[items_df["movie_title"].str.contains(search_query, case=False, na=False)]

        if not matching_movies.empty:
            st.write(f"Found {len(matching_movies)} matching movies:")
            for _, row in matching_movies.iterrows():
                display_movie(row["movie_title"], row["movie_id"], poster_dict)
        else:
            st.write("No movies found.")

