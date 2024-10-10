import numpy as np
import matplotlib.pyplot as plt

class KNearestNeighbour:
    # Non-parametric model.
    def __init__(self, x_data, y_data, k):
        self.x_data = x_data
        self.y_data = y_data
        self.k = k
        
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))
    
    def get_k_nearest_neighbours(self, distances, k):
        return np.argsort(distances)[:k]
    
    def majority_votes(self, neighbour_labels):
        values, counts = np.unique(neighbour_labels, return_counts=True)
        max_count = np.max(counts)
        tied_classes = values[counts == max_count]
        if len(tied_classes) > 1:
            return np.random.choice(tied_classes)
        else:
            return tied_classes[0]
    
    def predict(self, x):
        distances  = self.euclidean_distance(self.x_data, x)
        neighbour_indices = self.get_k_nearest_neighbours(distances, self.k)
        neighbour_labels = self.y_data[neighbour_indices]
        return self.majority_votes(neighbour_labels)

np.random.seed(42)
x_data = np.random.rand(100, 2)
y_data = np.random.randint(0, 2, size=(100,))
x_test = np.random.rand(20, 2)

knn = KNearestNeighbour(x_data, y_data, k=5)
y_pred = [knn.predict(x) for x in x_test]
print(y_pred)

