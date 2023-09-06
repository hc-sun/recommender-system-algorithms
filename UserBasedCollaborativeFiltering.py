import numpy as np

class UserBasedCollaborativeFiltering:
    def __init__(self, ratings_matrix):
        self.ratings_matrix = ratings_matrix
        self.num_users, self.num_items = ratings_matrix.shape
        self.mean_ratings = np.nanmean(ratings_matrix, axis=1)
        self.normalized_ratings = ratings_matrix - self.mean_ratings[:, np.newaxis]

    def pearson_similarity(self, user1, user2):
        common_items = np.logical_and(~np.isnan(self.ratings_matrix[user1]), ~np.isnan(self.ratings_matrix[user2]))
        if np.sum(common_items) == 0:
            return 0  #no common rated items
        normalized_user1 = self.normalized_ratings[user1, common_items]
        normalized_user2 = self.normalized_ratings[user2, common_items]
        norm_user1 = np.linalg.norm(normalized_user1)
        norm_user2 = np.linalg.norm(normalized_user2)
        
        if norm_user1 == 0 or norm_user2 == 0:
            return 0  #zero division
        else:
            similarity = np.dot(normalized_user1, normalized_user2) / (norm_user1 * norm_user2)
            return similarity

    def find_neighbors(self, target_user, k):
        similarities = [self.pearson_similarity(target_user, user) for user in range(self.num_users)]
        sorted_indices = np.argsort(similarities)[::-1]  #sort in descending order
        neighbors = [user for user in sorted_indices if user != target_user][:k]
        return neighbors

    def predict_rating(self, target_user, item, k):
        neighbors = self.find_neighbors(target_user, k)
        numerator = 0
        denominator = 0
        for neighbor in neighbors:
            if not np.isnan(self.ratings_matrix[neighbor, item]):
                similarity = self.pearson_similarity(target_user, neighbor)
                numerator += similarity * (self.ratings_matrix[neighbor, item] - self.mean_ratings[neighbor])
                denominator += np.abs(similarity)
        if denominator == 0:
            return self.mean_ratings[target_user]
        predicted_rating = self.mean_ratings[target_user] + (numerator / denominator)
        return predicted_rating
