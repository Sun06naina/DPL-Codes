import numpy as np
import matplotlib.pyplot as plt

# Input dataset (fixed 0 instead of O)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [1]])

print("OR GATE SINGLE PERCEPTRON - HARD-CODED WEIGHTS")
print("=" * 60)

# Weights and bias
weights = np.array([[1.0], [1.0]])
bias = np.array([[-0.5]])

print(f"Weights: W1={weights[0,0]}, W2={weights[1,0]}")
print(f"Bias: b={bias[0,0]}")
print("Equation: 1*x1 + 1*x2 - 0.5 > 0")

# Step function
def step_function(x):
    return 1 if x > 0 else 0

# Perceptron prediction
def perceptron_predict(X, weights, bias):
    linear = np.dot(X, weights) + bias
    predictions = np.zeros((len(X), 1))
    
    for i in range(len(X)):
        predictions[i, 0] = step_function(linear[i, 0])
    
    return predictions

# Predictions
print("\nPREDICTIONS")
linear_values = np.dot(X, weights) + bias
predictions = perceptron_predict(X, weights, bias)

for i in range(len(X)):
    print(f"{X[i]} → Linear: {linear_values[i,0]:.1f} → Pred: {int(predictions[i,0])} | Actual: {y[i,0]}")

# Plotting
plt.figure(figsize=(12, 5))

# Scatter plot
plt.subplot(1, 2, 1)
plt.scatter(X[y.flatten()==0, 0], X[y.flatten()==0, 1], c='red', s=200, label='Class 0')
plt.scatter(X[y.flatten()==1, 0], X[y.flatten()==1, 1], c='blue', s=200, label='Class 1')

# Decision boundary: x2 = 0.5 - x1
x1 = np.linspace(-0.5, 1.5, 100)
x2 = 0.5 - x1
plt.plot(x1, x2, 'g', label='Decision Boundary')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('OR Gate Decision Boundary')
plt.legend()
plt.grid()

# Bar graph
plt.subplot(1, 2, 2)
labels = ['[0,0]', '[0,1]', '[1,0]', '[1,1]']
values = linear_values.flatten()

plt.bar(labels, values)
plt.axhline(0, linestyle='--')
plt.title("Linear Values")
plt.grid()

plt.tight_layout()
plt.show()

# Verification
print("\nVERIFICATION RESULTS")
for i in range(4):
    z = X[i,0] + X[i,1] - 0.5
    print(f"{X[i]} → {z:.1f} → {'Class 1' if z > 0 else 'Class 0'}")