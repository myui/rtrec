import os
import streamlit as st
import pandas as pd
import requests
import re
import urllib.parse

OMDB_API_KEY = os.getenv("OMDB_API_KEY")

# URLs for MovieLens 100K dataset
DATASET_URL = "http://files.grouplens.org/datasets/movielens/ml-100k/"
RATINGS_FILE = DATASET_URL + "u.data"
ITEMS_FILE = DATASET_URL + "u.item"

# Caching functions
@st.cache_data
def load_movielens_100k():
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
def load_movie_posters():
    """Load movie posters from an external source."""
    poster_url = "https://raw.githubusercontent.com/babu-thomas/movielens-posters/refs/heads/master/movie_poster.csv"
    poster_df = pd.read_csv(poster_url, header=None, names=['movie_id', 'url'])
    return dict(zip(poster_df['movie_id'], poster_df['url']))

def fetch_movie_details(title):
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
                return data.get("Poster"), data.get("Plot")
            return None, data.get("Error", "No description available.")
    except requests.RequestException:
        return None, "Error fetching movie details."

    return None, "No description available."

def recommend_movies(user_id, df, feedback, top_n=5):
    """Generate movie recommendations excluding already seen movies."""
    seen_movies = df[df["user_id"] == user_id]["movie_id"].unique()
    recommendations = df[~df["movie_id"].isin(seen_movies)].drop_duplicates("movie_id")
    return recommendations.sample(n=top_n)

def display_movie(title, movie_id, poster_dict):
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
    def handle_feedback():
        selected_index = feedback_options.index(st.session_state[feedback_id])
        if selected_index == 0:
            st.session_state.feedback["liked"].append(title)
        elif selected_index == 1 and title not in st.session_state.feedback["disliked"]:
            st.session_state.feedback["disliked"].append(title)

    st.radio("Feedback", options=feedback_options, key=feedback_id, index=None, horizontal=True, on_change=handle_feedback)

# Streamlit UI
st.title("MovieLens 100K Recommender System")
st.sidebar.header("User Input")

# Load data
ratings_df, items_df, merged_df = load_movielens_100k()
poster_dict = load_movie_posters()

# User input
user_id = st.sidebar.number_input("Enter User ID", min_value=1, value=1)
top_n = st.sidebar.slider("Number of Recommendations", min_value=1, max_value=10, value=5)
recent_feedback_n = st.sidebar.slider("Recent Feedback Items (N)", min_value=1, max_value=20, value=5)

# Initialize session state
if "feedback" not in st.session_state:
    st.session_state.feedback = {"liked": [], "disliked": []}

# Tabs for navigation
tab1, tab2, tab3 = st.tabs(["Recommendations", "Recently Feedbacked Movies", "Search Movies"])

with tab1:
    # Get recommendations
    recommendations = recommend_movies(user_id, merged_df, st.session_state.feedback, top_n)
    st.write(f"Top {top_n} recommendations for User {user_id}:")

    for _, row in recommendations.iterrows():
        display_movie(row["movie_title"], row["movie_id"], poster_dict)

with tab2:
    st.write("### Recently Feedbacked Movies")
    liked_movies = st.session_state.feedback["liked"][-recent_feedback_n:]
    disliked_movies = st.session_state.feedback["disliked"][-recent_feedback_n:]

    if liked_movies or disliked_movies:
        if liked_movies:
            st.write("#### Liked Movies")
            for movie in liked_movies:
                st.write(f"👍 {movie}")

        if disliked_movies:
            st.write("#### Disliked Movies")
            for movie in disliked_movies:
                st.write(f"👎 {movie}")
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

# Display raw data
if st.sidebar.checkbox("Show Raw Data"):
    st.write("Ratings Data", ratings_df.head())
    st.write("Items Data", items_df.head())
