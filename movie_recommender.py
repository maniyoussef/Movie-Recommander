import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity
import os
from typing import Dict, List
from functools import lru_cache


class MovieRecommender:
    def __init__(self):
        self.data_df = None
        self.user_movie_matrix = None
        self.similarity_matrix = None
        self.api_key = "9aa76956e23338eaa84ce6a14dd54907"  # TMDb API Key
        self.load_data()

    @lru_cache(maxsize=512)
    def fetch_poster(self, title):
        """Fetch movie poster URL from TMDb API."""
        base_url = "https://api.themoviedb.org/3/search/movie"
        params = {
            "api_key": self.api_key,
            "query": title
        }
        try:
            response = requests.get(base_url, params=params, timeout=5)
            if response.status_code == 200:
                results = response.json().get("results")
                if results:
                    poster_path = results[0].get("poster_path")
                    if poster_path:
                        return f"https://image.tmdb.org/t/p/w500{poster_path}"
        except Exception as e:
            print(f"Error fetching poster for {title}: {e}")
        return "https://via.placeholder.com/500x750?text=No+Image"

    def load_data(self):
        """Load data from the specified CSV files."""
        credits_path = r"C:\Users\youss\OneDrive\Bureau\MovieDataset\tmdb_5000_credits.csv"
        movies_path = r"C:\Users\youss\OneDrive\Bureau\MovieDataset\tmdb_5000_movies.csv"

        if not os.path.exists(credits_path) or not os.path.exists(movies_path):
            st.error("❌ Dataset files not found. Please check the paths.")
            return

        credits_df = pd.read_csv(credits_path)
        movies_df = pd.read_csv(movies_path)

        merged_df = movies_df.merge(credits_df, left_on='id', right_on='movie_id')

        # Rename and ensure required columns
        merged_df.rename(columns={
            'title_x': 'title',
            'vote_average': 'vote_average',
            'popularity': 'popularity',
            'release_date': 'release_date'
        }, inplace=True)

        required_columns = ['title', 'vote_average', 'release_date', 'popularity']
        self.data_df = merged_df[required_columns]

        self.prepare_data()

    def prepare_data(self):
        """Prepare the data for recommendation calculations."""
        if self.data_df is not None:
            n_synthetic_users = 100
            ratings = self.data_df['vote_average'].values / 2

            np.random.seed(42)
            synthetic_ratings = np.zeros((n_synthetic_users, len(self.data_df)))

            for i in range(n_synthetic_users):
                noise = np.random.normal(0, 0.5, len(ratings))
                synthetic_ratings[i] = np.clip(ratings + noise, 1, 5)

            self.user_movie_matrix = pd.DataFrame(
                synthetic_ratings,
                columns=self.data_df['title']
            )

            self.similarity_matrix = cosine_similarity(self.user_movie_matrix)

    def get_movie_recommendations(self, selected_movies: List[str], n_recommendations: int = 5) -> List[Dict]:
        """Get movie recommendations based on a list of movie titles."""
        if self.data_df is None or not selected_movies:
            return []

        selected_indices = self.data_df[self.data_df['title'].isin(selected_movies)].index
        similar_scores = {}
        for movie_idx in selected_indices:
            for idx, row in self.data_df.iterrows():
                if idx not in selected_indices:
                    score = (
                        abs(self.data_df.loc[movie_idx, 'vote_average'] - row['vote_average']) +
                        abs(self.data_df.loc[movie_idx, 'popularity'] - row['popularity'])
                    )
                    if idx in similar_scores:
                        similar_scores[idx] += score
                    else:
                        similar_scores[idx] = score

        sorted_scores = sorted(similar_scores.items(), key=lambda x: x[1])
        recommendations = []
        for idx, score in sorted_scores[:n_recommendations]:
            movie = self.data_df.iloc[idx]
            recommendations.append({
                'title': movie['title'],
                'vote_average': round(movie['vote_average'], 2),
                'popularity': round(movie['popularity'], 2),
                'release_date': movie['release_date'],
                'poster_path': self.fetch_poster(movie['title'])
            })

        return recommendations


def create_movie_stats(recommender: MovieRecommender) -> Dict:
    """Create statistics about the movie dataset."""
    if recommender.data_df is None:
        return {
            'Total Movies': 0,
            'Average Rating': 0,
            'Average Popularity': 0
        }

    stats = {
        'Total Movies': len(recommender.data_df),
        'Average Rating': round(recommender.data_df['vote_average'].mean(), 2),
        'Average Popularity': round(recommender.data_df['popularity'].mean(), 2)
    }
    return stats


def main():
    st.set_page_config(
        page_title="Movie Recommender",
        page_icon="🎥",
        layout="wide"
    )

    st.markdown("<h1 style='text-align:center;'>🎥 Movie Recommendation System</h1>", unsafe_allow_html=True)

    @st.cache_resource
    def load_recommender():
        return MovieRecommender()

    recommender = load_recommender()

    if recommender.data_df is not None:
        st.sidebar.header("📊 Dataset Statistics")
        stats = create_movie_stats(recommender)
        for key, value in stats.items():
            st.sidebar.metric(key, value)

        st.write("### Get Movie Recommendations")

        movie_titles = recommender.data_df['title'].tolist()
        selected_movies = st.multiselect(
            "Select one or more movies",
            options=movie_titles,
            help="Choose your favorite movies for personalized recommendations."
        )

        n_recommendations = st.slider(
            "Number of Recommendations",
            1, 20, 5
        )

        if st.button("🎬 Get Recommendations"):
            with st.spinner("Fetching recommendations..."):
                recommendations = recommender.get_movie_recommendations(
                    selected_movies,
                    n_recommendations
                )

            if recommendations:
                st.subheader(f"🎯 Top {n_recommendations} Movie Recommendations")
                for movie in recommendations:
                    st.markdown(
                        f"""
                        <div style='margin-bottom:20px;'>
                            <img src='{movie['poster_path']}' width='100' align='left' style='margin-right:15px;'/>
                            <h3>{movie['title']}</h3>
                            <p><b>📅 Release Date:</b> {movie['release_date']}</p>
                            <p><b>⭐ Rating:</b> {movie['vote_average']}/10</p>
                            <p><b>🔥 Popularity:</b> {movie['popularity']}</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.error("❌ No recommendations could be generated.")
    else:
        st.error("Failed to load the datasets. Please check the file paths and format.")


if __name__ == "__main__":
    main()
