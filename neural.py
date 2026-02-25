import numpy as np

class Perceptron:
    def __init__(self, inputSize, hiddenSizes, outputSize,hiddenSizesS):
        
        self.Win = np.zeros((1+inputSize,hiddenSizes))
        self.Win[0,:] = (np.random.randint(0, 3, size = (hiddenSizes)))
        self.Win[1:,:] = (np.random.randint(-1, 2, size = (inputSize,hiddenSizes)))
        
        self.Win1 = np.zeros((1 + hiddenSizes, hiddenSizesS))
        self.Win1[0, :] = np.random.randint(0, 3, size=hiddenSizesS)
        self.Win1[1:, :] = np.random.randint(-1, 2, size=(hiddenSizes, hiddenSizesS))
        
        self.Wout = np.random.randint(0, 2, size = (1+hiddenSizes,outputSize)).astype(np.float64)
        #self.Wout = np.random.randint(0, 3, size = (1+hiddenSizes,outputSize))
        
    def predict(self, Xp):
        # Первый скрытый слой
        net1 = np.dot(Xp, self.Win[1:, :]) + self.Win[0, :]
        hidden1 = np.where(net1 >= 0.0, 1, -1).astype(np.float64)

        # Второй скрытый слой
        net2 = np.dot(hidden1, self.Win1[1:, :]) + self.Win1[0, :]
        hidden2 = np.where(net2 >= 0.0, 1, -1).astype(np.float64)

        # Выходной слой
        net3 = np.dot(hidden2, self.Wout[1:, :]) + self.Wout[0, :]
        out = np.where(net3 >= 0.0, 1, -1).astype(np.float64)

        return out, hidden2

    def train(self, X, y, n_iter=5, eta = 0.01):
        for epoch in range(n_iter):
            print(self.Wout.reshape(1, -1))   
            for xi, target in zip(X, y):
                pr, hidden2 = self.predict(xi)
                error = target - pr

               
                self.Wout[1:] += (eta * error * hidden2).reshape(-1, 1)
                self.Wout[0] += eta * error

        return self
