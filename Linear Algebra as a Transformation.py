import matplotlib.pyplot as plt
import numpy as np 

v = np.array([[0,1],
              [0,2]])
print(v)
plt.plot(*v)
plt.show()

identity_matix = np.array([[1,0],
                           [0,1]])

plt.plot(*np.dot(v, identity_matix))
plt.show()