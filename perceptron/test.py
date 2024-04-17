import numpy as np
import matplotlib.pyplot as plt


def generate_linearly_separable_data(num_samples, num_features=2, separation_distance=1.0):
    # Generate random weights for separating line
    weights = np.random.randn(num_features)

    # Generate random points
    X = np.random.randn(num_samples, num_features)

    # Compute labels based on whether points are above or below the separating line
    y = np.sign(np.dot(X, weights))

    # Add separation distance to labels to ensure classes are well-separated
    X = separation_distance

    return X, y


# Generate linearly separable data
num_samples = 100
X, y = generate_linearly_separable_data(num_samples, 3, 1)

# Plot the data
plt.scatter(X[:, 0], X[:, 2], c=y, cmap=plt.cm.bwr, edgecolors='k')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Linearly Separable Data')
plt.show()
