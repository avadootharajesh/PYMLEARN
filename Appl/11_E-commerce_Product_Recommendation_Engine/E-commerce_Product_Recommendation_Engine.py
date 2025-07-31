# E-commerce_Product_Recommendation_Engine.py
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def load_sample_data():
    # Sample user-product ratings
    data = {
        'UserID': ['U1', 'U1', 'U1', 'U2', 'U2', 'U3', 'U3', 'U4', 'U4', 'U5'],
        'ProductID': ['P1', 'P2', 'P3', 'P1', 'P4', 'P2', 'P3', 'P1', 'P4', 'P3'],
        'Rating': [5, 3, 4, 4, 5, 2, 5, 5, 4, 3]
    }
    df = pd.DataFrame(data)
    return df

def create_user_item_matrix(df):
    user_item = df.pivot(index='ProductID', columns='UserID', values='Rating').fillna(0)
    return user_item

def compute_item_similarity(user_item):
    similarity = cosine_similarity(user_item)
    similarity_df = pd.DataFrame(similarity, index=user_item.index, columns=user_item.index)
    return similarity_df

def recommend_products(product_id, similarity_df, user_item, top_n=3):
    if product_id not in similarity_df.columns:
        print(f"Product {product_id} not found in data.")
        return []

    sim_scores = similarity_df[product_id].drop(product_id)
    top_similar = sim_scores.sort_values(ascending=False).head(top_n)
    recommendations = top_similar.index.tolist()
    return recommendations

def main():
    df = load_sample_data()
    user_item = create_user_item_matrix(df)
    similarity_df = compute_item_similarity(user_item)

    product_to_recommend = 'P1'
    recommendations = recommend_products(product_to_recommend, similarity_df, user_item)

    print(f"Products similar to {product_to_recommend}: {recommendations}")

if __name__ == "__main__":
    main()
