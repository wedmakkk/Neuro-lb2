import numpy as np

class Perceptron:
    def __init__(self, inputSize, hiddenSizes, hiddenSizesS, outputSize):
        # Веса первого скрытого слоя
        self.Win = np.zeros((1 + inputSize, hiddenSizes))
        self.Win[0, :] = np.random.randint(0, 3, size=hiddenSizes)
        self.Win[1:, :] = np.random.randint(-1, 2, size=(inputSize, hiddenSizes))

        # Веса второго скрытого слоя
        self.Wins = np.zeros((1 + hiddenSizes, hiddenSizesS))
        self.Wins[0, :] = np.random.randint(0, 3, size=hiddenSizesS)
        self.Wins[1:, :] = np.random.randint(-1, 2, size=(hiddenSizes, hiddenSizesS))

        # Веса выходного слоя (размерность соответствует второму скрытому слою)
        self.Wout = np.random.randint(0, 2, size=(1 + hiddenSizesS, outputSize)).astype(np.float64)

    def predict(self, Xp):
        # Первый скрытый слой
        net1 = np.dot(Xp, self.Win[1:, :]) + self.Win[0, :]
        hidden1 = np.where(net1 >= 0.0, 1, -1).astype(np.float64)

        # Второй скрытый слой
        net2 = np.dot(hidden1, self.Wins[1:, :]) + self.Wins[0, :]
        hidden2 = np.where(net2 >= 0.0, 1, -1).astype(np.float64)

        # Выходной слой
        net3 = np.dot(hidden2, self.Wout[1:, :]) + self.Wout[0, :]
        out = np.where(net3 >= 0.0, 1, -1).astype(np.float64)

        return out, hidden2

    def train(self, X, y, n_iter=100, eta=0.1):
        for epoch in range(n_iter):
            for xi, target in zip(X, y):
                pr, hidden2 = self.predict(xi)
                error = target - pr
                self.Wout[1:] += (eta * error * hidden2).reshape(-1, 1)
                self.Wout[0] += eta * error
        return self