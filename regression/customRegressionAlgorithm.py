from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
#Sample data
xs = np.array([1,2,3,4,5,6], dtype=np.float64)
ys = np.array([5,4,6,5,6,7], dtype=np.float64)

plt.scatter(xs,ys)
plt.show()
