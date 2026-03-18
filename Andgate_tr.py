import numpy as np
import matplotlib.pyplot as plt

# Parameters
input_size = 2
learning_rate = 0.01
iterations = 3000
print_interval = 1000

# Dataset (FIXED: 0 instead of O)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [0], [0], [1]])

# Initialize weights
np.random.seed(42)
weights = np.random.uniform(-1, 1, (input_size, 1))
bias = np.array([[0.0]])

print("Initial Weights and Bias:")
print(f"Weights: {weights.flatten()}")
print(f"Bias: {bias[0,0]:.3f}")

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

# Derivative
def sigmoid_derivative(x):
    return x * (1 - x)

# Prediction
def predict(X, weights, bias):
    return sigmoid(np.dot(X, weights) + bias)

# Training
print("\nTraining Single Perceptron (AND Gate)...")
loss_history = []

for iteration in range(iterations):
    linear_output = np.dot(X, weights) + bias
    output = sigmoid(linear_output)

    error = y - output
    d_output = error * sigmoid_derivative(output)

    # Update weights
    weights += learning_rate * np.dot(X.T, d_output)
    bias += learning_rate * np.sum(d_output, axis=0, keepdims=True)

    loss = np.mean(np.abs(error))
    loss_history.append(loss)

    if iteration % print_interval == 0:
        print(f"Iteration {iteration}: Loss = {loss:.4f}")

print("\nTraining Complete!")
print(f"Final Weights: W1={weights[0,0]:.3f}, W2={weights[1,0]:.3f}")
print(f"Final Bias: {bias[0,0]:.3f}")

# Predictions
final_predictions = predict(X, weights, bias)

print("\nPredictions:")
for i in range(len(X)):
    pred_class = 1 if final_predictions[i,0] > 0.5 else 0
    print(f"Input {X[i]} → {final_predictions[i,0]:.3f} → Class {pred_class}")

# Plot
plt.figure(figsize=(12,5))

# Scatter
plt.subplot(1,2,1)
plt.scatter(X[y.flatten()==0, 0], X[y.flatten()==0, 1], c='red', s=200, label='Class 0')
plt.scatter(X[y.flatten()==1, 0], X[y.flatten()==1, 1], c='blue', s=200, label='Class 1')

# Decision boundary
w1, w2 = weights[0,0], weights[1,0]
b = bias[0,0]
x1 = np.linspace(-0.2, 1.2, 100)

if abs(w2) > 1e-6:
    x2 = (-w1 * x1 - b) / w2
    plt.plot(x1, x2, 'g', label='Decision Boundary')
else:
    plt.axvline(x=-b/w1, color='g', label='Decision Boundary')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Decision Boundary')
plt.legend()
plt.grid()

# Loss graph
plt.subplot(1,2,2)
plt.plot(loss_history)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid()

plt.tight_layout()
plt.show()

# Verification
print("\nDecision Boundary Equation:")
print(f"{weights[0,0]:.3f}*x1 + {weights[1,0]:.3f}*x2 + {bias[0,0]:.3f} = 0")