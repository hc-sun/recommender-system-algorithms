import numpy as np
import pandas as pd
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error

class MatrixFactorization:
    def __init__(self, rating_matrix, latent_dim=10, learning_rate=0.01, num_iterations=5):
        self.rating_matrix = rating_matrix
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.num_users, self.num_items = rating_matrix.shape
        self.user_matrix = np.random.rand(self.num_users, latent_dim)
        self.item_matrix = np.random.rand(self.num_items, latent_dim)

    def fit(self):
        for iteration in range(self.num_iterations):
            for i in range(self.num_users):
                for j in range(self.num_items):
                    if np.isnan(self.rating_matrix[i, j]):
                        continue  #skip unobserved ratings
                    predicted_rating = np.dot(self.user_matrix[i], self.item_matrix[j])
                    error = self.rating_matrix[i, j] - predicted_rating
                    self.user_matrix[i] += self.learning_rate * (error * self.item_matrix[j])
                    self.item_matrix[j] += self.learning_rate * (error * self.user_matrix[i])
            print("Number of training iteration: {}".format(iteration))

    def predict(self):
        return np.dot(self.user_matrix, self.item_matrix.T)
    
    def evaluate(self):
        #mask for observed ratings
        test_mask = ~np.isnan(self.rating_matrix)
        test_actual = self.rating_matrix[test_mask]
        test_preds = self.predict()[test_mask]

        rmse = math.sqrt(mean_squared_error(test_actual, test_preds))
        mae = mean_absolute_error(test_actual, test_preds)

        return rmse, mae
