import numpy as np
from scipy.linalg import svd

class DimensionalityReductionCF:
    def __init__(self, ratings_matrix, dimensionality):
        self.ratings_matrix = ratings_matrix
        self.dimensionality = dimensionality
        self.num_users, self.num_items = ratings_matrix.shape
        self.mean_ratings = np.nanmean(ratings_matrix, axis=1, keepdims=True)
        self.filled_matrix = self.fill_missing_entries()

    def fill_missing_entries(self):
        filled_matrix = np.copy(self.ratings_matrix)
        for user_idx in range(self.num_users):
            nan_indices = np.isnan(filled_matrix[user_idx, :])
            filled_matrix[user_idx, nan_indices] = self.mean_ratings[user_idx]
        return filled_matrix

    def perform_svd(self):
        U, s, Vt = svd(self.filled_matrix, full_matrices=False)
        reduced_matrix = np.dot(U[:, :self.dimensionality], np.diag(s[:self.dimensionality]))
        return reduced_matrix
