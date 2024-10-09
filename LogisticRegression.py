import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, x_data, y_data):
        self.x_data = np.c_[np.ones((x_data.shape[0],1)), self.scale_feature(x_data)]
        self.y_data = y_data.reshape(-1,1)
        self.beta = np.random.randn(self.x_data.shape[1],1) * 0.01
    
    def fit(self, learning_rate, iterations):
        cost_history = self.gradient_descent(learning_rate, iterations)
        return cost_history
        
    def scale_feature(self, x):
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        return (x - mean) / std
    
    @staticmethod
    def _sigmoid(z):
        return 1/ (1 + np.exp(-z))
    
    def _compute_cost(self, x, y, beta):
        #cost function is BCE
        m = x.shape[0]
        prediction = self._sigmoid(x.dot(beta))
        cost = (-1/m) * np.sum(y * np.log(prediction) + (1-y) * np.log(1-prediction))
        return cost
    
    def gradient_descent(self, learning_rate, iterations):
        cost_history = []
        m = self.x_data.shape[0]
        for _ in range(iterations):
            prediction = self._sigmoid(self.x_data.dot(self.beta))
            gradient = 1/m * self.x_data.T.dot(prediction - self.y_data)
            self.beta -= learning_rate * gradient
            cost_history.append(self._compute_cost(self.x_data, self.y_data, self.beta))
        return cost_history
            
    def predict(self, x_data):
        x_data = np.c_[np.ones((x_data.shape[0], 1)), self.scale_feature(x_data)]
        return self._sigmoid(x_data.dot(self.beta)) > 0.5
            

def plot_cost_history(cost_history):
    iterations = len(cost_history)
    plt.plot(range(iterations), cost_history, color='orange')
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Cost Function Decreasing Over Time")
    plt.show()
    
    
def sigmoid(z):
    return 1/ (1 + np.exp(-z))

np.random.seed(42)
X = np.random.rand(1000, 2)
true_beta = np.array([3,5,2]).reshape(-1,1)    
linear_combination = X.dot(true_beta[1:]) + true_beta[0]
probabilities = sigmoid(linear_combination)
threshold = np.median(probabilities) 
y = (probabilities >= threshold).astype(int)

log_reg = LogisticRegression(X, y)
cost_history = log_reg.fit(learning_rate=0.01, iterations=10000)
print("Actual Coefficients:", true_beta.T)
print("Estimated Coefficients:", log_reg.beta.T)
plot_cost_history(cost_history)

        
