import random
import numpy as np
import matplotlib.pyplot as plt
from customRegressionAlgorithm import best_fit_slope_intercept, coefficient_of_correlation
from customRegressionAlgorithm import find_and_plot_regression

def create_dataset(num_data, variance, step=2):
	val = 1
	ys = []
	xs = []
	for i in range(num_data):
		y = val + random.randrange(-variance, variance)
		ys.append(y)
		val += step
	xs = [i for i in range(len(ys))]
	return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

xs, ys = create_dataset(40, 30, step=10)


print(coefficient_of_correlation(xs, ys, find_and_plot_regression(xs, ys)))