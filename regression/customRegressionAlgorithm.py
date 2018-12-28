from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
#Sample data
xs = np.array([1,2,3,4,5,6], dtype=np.float64)
ys = np.array([5,4,6,5,6,7], dtype=np.float64)

#plt.scatter(xs,ys)
#plt.show()

def best_fit_slope_intercept(xs, ys):
	numerator = (mean(xs)*mean(ys)) - mean(xs*ys)
	denominator = mean(xs)**2 - mean(xs**2)
	m = numerator / denominator
	b = mean(ys) - m*mean(xs)
	return m, b
	#m = ( ((mean(xs) * mean(ys)) - mean(xs*ys)) /
	#		(mean(xs)*mean(xs)) - mean(xs*xs))
	#return m

m, b = best_fit_slope_intercept(xs, ys)

regression_line = [(m*x) + b for x in xs]
#plt.scatter(xs, ys)
#plt.plot(xs, regression_line)
#plt.show()

#Let's try to calculate r^2 before watching the tutorial for it
def coefficient_of_correlation(xs, ys, regression_line):
	#Calculate the squared error of the regression line
	se_regression = 0
	for c, y_hat in enumerate(regression_line):
		se_regression += (y_hat - ys[c])**2
	#Calculate the squared error of the mean of the y's
	se_mean_ys = 0
	for c in ys:
		se_mean_ys += (c - mean(ys))**2
	return 1 - (se_regression/se_mean_ys)

print(coefficient_of_correlation(xs,ys,regression_line))