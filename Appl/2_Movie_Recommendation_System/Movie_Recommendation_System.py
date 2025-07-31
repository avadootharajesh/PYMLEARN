# Movie_Recommendation_System.py
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def load_data():
    # Using MovieLens 100k dataset sample (small and easy to work with)
    # Download from: https://files.grouplens.org/datasets/movielens/ml-100k/u.data
    # Movie titles: https://files.grouplens.org/datasets/movielens/ml-100k/u.item
    ratings_url = "https://raw.githubusercontent.com/sidooms/MovieTweetings/master/latest/ratings.dat"
    # Note: For demo, use a small subset or local copy
    # We'll simulate small ratings data below
    
    # For demonstration, create a small sample ratings DataFrame:
    data = {'userId': [1, 1, 1, 2, 2, 3, 3, 4],
            'movieId': [10, 20, 30, 10, 40, 20, 30, 40],
            'rating': [4, 5, 3, 4, 2, 5, 4, 3]}
    df = pd.DataFrame(data)
    return df

def create_user_item_matrix(df):
    user_item = df.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
    return user_item

def compute_user_similarity(user_item):
    # Cosine similarity between users
    similarity = cosine_similarity(user_item)
    similarity_df = pd.DataFrame(similarity, index=user_item.index, columns=user_item.index)
    return similarity_df

def get_recommendations(user_id, user_item, similarity_df, top_n=3):
    # Weighted sum of ratings from similar users
    user_sim_scores = similarity_df[user_id].drop(user_id)
    similar_users = user_sim_scores[user_sim_scores > 0].index
    
    # Compute weighted ratings
    weighted_ratings = pd.Series(dtype=float)
    for other_user in similar_users:
        sim_score = user_sim_scores[other_user]
        other_user_ratings = user_item.loc[other_user]
        weighted_ratings = weighted_ratings.add(other_user_ratings * sim_score, fill_value=0)
    
    # Normalize by sum of similarities
    sum_sim = user_sim_scores[similar_users].sum()
    if sum_sim > 0:
        weighted_ratings /= sum_sim
    else:
        weighted_ratings = pd.Series(dtype=float)
    
    # Remove movies the user has already rated
    user_rated = user_item.loc[user_id]
    weighted_ratings = weighted_ratings[user_rated == 0]
    
    # Get top N recommendations
    recommendations = weighted_ratings.sort_values(ascending=False).head(top_n)
    return recommendations

def main():
    df = load_data()
    user_item = create_user_item_matrix(df)
    similarity_df = compute_user_similarity(user_item)
    
    user_id = 1
    recommendations = get_recommendations(user_id, user_item, similarity_df)
    
    print(f"Top recommendations for user {user_id}:\n{recommendations}")

if __name__ == "__main__":
    main()
