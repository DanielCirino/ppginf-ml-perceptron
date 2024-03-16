import numpy as np


class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs=10):
        self.weights = np.zeros(input_size + 1)
        self.learning_rate = learning_rate
        self.epochs = epochs

    def gerar_dataset(self, qtdRegistros=100, qtdAtributos=2, distancia=1):
        # Gerar os pesos para cada atributo (weights)
        pesos = np.random.randn(qtdAtributos)

        # gerar pontos de dados aleatórios (data points) aplicando a distância
        # para metade dos dados é somada a distância e para outra metade é subtraída
        x1 = np.random.randn(qtdRegistros//2, qtdAtributos)+distancia
        x2 = np.random.randn(qtdRegistros//2, qtdAtributos)-distancia
        X = np.vstack((x1,x2))

        # Gerar os rótulos para cada ponto de dados (labels)
        # Os rótulos são 1 e -1
        y = np.sign(np.dot(X, pesos))
        return X, y

    def fazer_ativacao(self, x):
        return 1 if x >= 0 else -1

    def prever(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return self.fazer_ativacao(summation)

    def treinar(self, training_inputs, labels):
        for _ in range(self.epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.prever(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)
