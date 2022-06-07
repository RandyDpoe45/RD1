import numpy as np
import matplotlib.pyplot as plt

#Example data
x = [1,2,3,4]
y = [0.5,0.61,0.75,0.4]
x = np.arange(0,4)
y = np.array(y)

#Fit line
slope, intercept = np.polyfit(x, y, 1)
print(slope, intercept)

#Plot
plt.figure()
plt.scatter(x, y)
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), color = 'k')
plt.show()

print(slope)