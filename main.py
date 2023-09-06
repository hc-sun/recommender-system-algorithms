import numpy as np
import pandas as pd
from MatrixFactorization import MatrixFactorization
from UserBasedCollaborativeFiltering import UserBasedCollaborativeFiltering
from ItemBasedCollaborativeFiltering import ItemBasedCollaborativeFiltering
from DimensionalityReductionCF import DimensionalityReductionCF

def recommend_items(user_id, predicted_ratings, N=5):
    user_ratings = predicted_ratings[user_id]
    top_indices = np.argsort(user_ratings)[::-1][:N]
    return top_indices

def main():
    ratings = pd.read_csv(r".\datasets\ratings.csv")
    rating_matrix = ratings.pivot(index='userId', columns='itemId', values='rating').fillna(np.nan).values


    #Example usage of Matrix Factorization Model
    mf_model = MatrixFactorization(rating_matrix)
    mf_model.fit()

    rmse, mae = mf_model.evaluate()
    print(f"Matrix Factorization RMSE: {rmse}")
    print(f"Matrix Factorization MAE: {mae}")

    #Example recommendation for user with ID 1
    user_id = 1
    recommendations = recommend_items(user_id, mf_model.predict(), N=5)
    print(f"Recommended items for user {user_id}: {recommendations}")

    #Example usage of User-Based Collaborative Filtering
    user_cf = UserBasedCollaborativeFiltering(rating_matrix)
    target_user = 2
    item_to_predict = 0
    k_neighbors = 2
    predicted_rating = user_cf.predict_rating(target_user, item_to_predict, k_neighbors)
    print(f"User-Based CF Predicted rating for item {item_to_predict} by user {target_user}: {predicted_rating}")


    #Example usage of Item-Based Collaborative Filtering
    item_cf = ItemBasedCollaborativeFiltering(rating_matrix)
    target_user = 2
    target_item = 0
    k_similar_items = 2
    predicted_rating = item_cf.predict_rating(target_user, target_item, k_similar_items)
    print(f"Item-Based CF Predicted rating for item {target_item} by user {target_user}: {predicted_rating}")


    #Example usage of Dimensionality Reduction CF
    dimensionality = 2
    dr_cf = DimensionalityReductionCF(rating_matrix, dimensionality)
    reduced_matrix = dr_cf.perform_svd()
    print(f"Dimensionality Reduction (SVD) Result: \n {reduced_matrix}")


if __name__ == "__main__":
    main()
