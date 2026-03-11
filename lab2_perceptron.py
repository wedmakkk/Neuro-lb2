# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 20:24:56 2021
@author: AM4
"""
import pandas as pd
import numpy as np
from neural import Perceptron  # импортируем класс из neural.py

# Загружаем и подготавливаем данные
df = pd.read_csv('datka.csv')
df = df.iloc[np.random.permutation(len(df))]
y = df.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", 1, -1)
X = df.iloc[0:100, [0, 2]].values

# Параметры сети
inputSize = X.shape[1]          # количество входных сигналов (2)
hiddenSizes = 10                 # число нейронов первого скрытого слоя
hiddenSizesS = 5                 # число нейронов второго скрытого слоя (можно изменить)
outputSize = 1                    # один выходной нейрон (биполярный)

# Создаём экземпляр перцептрона
p = Perceptron(inputSize, hiddenSizes, hiddenSizesS, outputSize)

# Обучаем сеть
p.train(X, y, n_iter=100, eta=0.1)

# Проверяем на обучающей выборке (первые 100 образцов)
pr, _ = p.predict(X)
errors = np.sum(pr.flatten() != y)
print(f"Ошибок на обучающей выборке (первые 100): {errors} из {len(y)}")

# Проверяем на всей выборке (все 150 образцов)
y_all = df.iloc[:, 4].values
y_all = np.where(y_all == "Iris-setosa", 1, -1)
X_all = df.iloc[:, [0, 2]].values
pr_all, _ = p.predict(X_all)
errors_all = np.sum(pr_all.flatten() != y_all)
print(f"Ошибок на полной выборке (150 образцов): {errors_all} из {len(y_all)}")