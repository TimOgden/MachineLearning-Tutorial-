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
print(m, b)
