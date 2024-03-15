import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs=100):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.zeros(input_size + 1)  # +1 para o viés
        self.errors = []

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]  # Adiciona o viés
        return self.activation(summation)

    def train(self, training_inputs, labels):
        for _ in range(self.epochs):
            errors = 0
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                errors += int(error != 0)
                self.weights[1:] += self.learning_rate * error * inputs
                self.weights[0] += self.learning_rate * error  # Atualiza o viés
            self.errors.append(errors)

def plot_training_progress(epochs, errors):
    plt.plot(range(1, epochs + 1), errors, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of Errors')
    plt.title('Training Progress')
    plt.show()

# Conjunto de dados de treinamento (AND gate)
training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = np.array([0, 0, 0, 1])

# Criação e treinamento do perceptron
perceptron = Perceptron(input_size=2)
perceptron.train(training_inputs, labels)

# Plotagem do progresso do treinamento
plot_training_progress(perceptron.epochs, perceptron.errors)
