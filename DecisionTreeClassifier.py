import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class DecisionTree:
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data.flatten()
    
    def build_tree(self, X, y, depth, max_depth=None):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        #Stopping Criteria
        if (max_depth is not None and depth >= max_depth) or n_classes ==1 or n_samples < 2:
            return {'class': np.bincount(y).argmax()}
        
        best_feature, best_threshold = self.find_best_split(X, y)
        
        X_left, y_left, X_right, y_right = self.split_data(X, y, best_feature, best_threshold)
        left_subtree = self.build_tree(X_left, y_left, depth+1, max_depth)
        right_subtree = self.build_tree(X_right, y_right, depth+1, max_depth)
        
        node = {
            'feature_index': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree
        }
        return node
        
    
    def find_best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        for feature_index in range(X.shape[1]):
            thresholds  = np.unique(X[:, feature_index])
            for threshold in thresholds:
                gain = self.information_gain(X, y, feature_index, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = threshold
        return best_feature, best_threshold
    
    def split_data(self, X, y, feature_index, threshold):
        left_indices = X[:, feature_index] <= threshold
        right_indices = X[:, feature_index] > threshold
        return X[left_indices], y[left_indices], X[right_indices], y[right_indices]
        
    def gini(self, y):
        y = y.flatten()
        p = np.bincount(y) / len(y)
        return 1 - np.sum(p **2)
    
    def information_gain(self, X, y, feature_index, threshold):
        n = len(y)
        parent_gain = self.gini(y)
        X_left, y_left, X_right, y_right = self.split_data(X, y, feature_index, threshold)
        weighted_avg_impurity = (len(y_left)/ n) * self.gini(y_left) + (len(y_right)/ n) * self.gini(y_right)
        return parent_gain - weighted_avg_impurity
    
    def fit(self, max_depth=None):
        self.tree = self.build_tree(self.X_data, self.y_data, depth=0, max_depth=max_depth)
        
    def predict_sample(self, x, tree=None):
        if tree is None:
            tree = self.tree
        
        if 'class' in tree:
            return tree['class']
            
        feature_value = x[tree['feature_index']]
        branch = tree['left'] if feature_value <= tree['threshold'] else tree['right']
        return self.predict_sample(x, branch)
        
    def predict(self, X):
        return np.array([self.predict_sample(x) for x in X])
        
        
np.random.seed(32)
X = np.random.randn(100, 2)
true_beta = np.array([3, 5]).reshape(-1, 1)
y = (np.dot(X, true_beta) + 2).flatten() > 0
y = y.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
decision_tree = DecisionTree(X_train, y_train)
decision_tree.fit(max_depth=3)

y_pred = decision_tree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)



