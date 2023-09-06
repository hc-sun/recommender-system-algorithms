import numpy as np

class ItemBasedCollaborativeFiltering:
    def __init__(self, ratings_matrix):
        self.ratings_matrix = ratings_matrix
        self.num_users, self.num_items = ratings_matrix.shape
        self.mean_ratings = np.nanmean(ratings_matrix, axis=1, keepdims=True)
        self.normalized_ratings = ratings_matrix - self.mean_ratings

    def adjusted_cosine_similarity(self, item1, item2):
        common_users = np.logical_and(~np.isnan(self.ratings_matrix[:, item1]), ~np.isnan(self.ratings_matrix[:, item2]))
        if np.sum(common_users) == 0:
            return 0  #no common users who rated both items
        normalized_item1 = self.normalized_ratings[common_users, item1]
        normalized_item2 = self.normalized_ratings[common_users, item2]
        similarity = np.dot(normalized_item1, normalized_item2) / (np.linalg.norm(normalized_item1) * np.linalg.norm(normalized_item2))
        return similarity

    def find_similar_items(self, target_item, k):
        similarities = [self.adjusted_cosine_similarity(target_item, item) for item in range(self.num_items)]
        sorted_indices = np.argsort(similarities)[::-1]  #sort in descending order
        similar_items = [item for item in sorted_indices if item != target_item][:k]
        return similar_items

    def predict_rating(self, target_user, target_item, k):
        similar_items = self.find_similar_items(target_item, k)
        numerator = 0
        denominator = 0
        for item in similar_items:
            if not np.isnan(self.ratings_matrix[target_user, item]):
                similarity = self.adjusted_cosine_similarity(target_item, item)
                numerator += similarity * self.ratings_matrix[target_user, item]
                denominator += np.abs(similarity)
        if denominator == 0:
            return self.mean_ratings[target_user][0]
        predicted_rating = numerator / denominator
        return predicted_rating
