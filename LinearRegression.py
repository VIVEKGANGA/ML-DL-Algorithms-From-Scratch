import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, x_data, y_data):
        self.x_data = np.c_[np.ones((x_data.shape[0],1)), x_data]
        self.y_data = y_data.reshape(-1, 1)
        self.beta = np.zeros((self.x_data.shape[1],1))
    
    def fit(self, learning_rate, iterations):
        cost_history = self.gradient_descent(learning_rate, iterations)
        return cost_history
        
    def predict(self, x):
        x = np.c_[np.ones((x.shape[0], 1)), x]
        return x.dot(self.beta)
    
    @staticmethod
    def _compute_cost(x, y, beta):
        #cost function is MSE
        m = x.shape[0]
        predictions = x.dot(beta)
        cost = (1/(2*m)) * np.sum(np.square(predictions - y))
        return cost
    
    def gradient_descent(self, learning_rate, iterations):
        m = self.x_data.shape[0]
        cost_history = []
        for _ in range(iterations):
            predictions = self.x_data.dot(self.beta)
            gradient = (1/m) * self.x_data.T.dot(predictions - self.y_data)
            self.beta -= learning_rate * gradient
            cost_history.append(self._compute_cost(self.x_data, self.y_data, self.beta))
        return cost_history

def plot_cost_history(cost_history):
    iterations = len(cost_history)
    plt.plot(range(iterations), cost_history, color='orange')
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Cost Function Decreasing Over Time")
    plt.show()
    
np.random.seed(42)
X = np.random.rand(100,2)
true_beta = np.array([5, 3, 2]).reshape(-1,1)
# y = 3x1 + 2x2+ 5
y = X.dot(true_beta[1:]) + true_beta[0] + np.random.normal(0, 1, size=(100, 1))

lr = LinearRegression(X, y)
cost_history = lr.fit(0.01, 1000)
print("Actual Coefficients:", true_beta.T)
print("Estimated Coefficients:", lr.beta.T)
plot_cost_history(cost_history)