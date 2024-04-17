import numpy as np
import matplotlib.pyplot as plt

from perceptron_core.perceptron import Perceptron


def plot_decision_boundary(X, y, perceptron):
    # Plot data points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.bwr, edgecolors='k')

    # Plot decision boundary
    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()
    x_values = np.linspace(xlim[0], xlim[1], 100)
    y_values = np.linspace(ylim[0], ylim[1], 100)
    xx, yy = np.meshgrid(x_values, y_values)
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = np.array([perceptron.prever(point) for point in grid_points])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, levels=[0.5], colors='k')

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Perceptron Decision Boundary')
    plt.show()





# Example data
# X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# y = np.array([0, 1, 1, 1])

# Train perceptron_core
perceptron = Perceptron(input_size=2)

X,y=perceptron.gerar_dataset(qtdRegistros=10,qtdAtributos=2,distancia=1.75)
perceptron.treinar(X, y)

# Plot decision boundary
plot_decision_boundary(X, y, perceptron)
