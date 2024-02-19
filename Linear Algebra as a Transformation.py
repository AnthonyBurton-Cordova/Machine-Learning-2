import matplotlib.pyplot as plt
import numpy as np 

# Create a numpy array 
v = np.array([[0,1],
              [0,2]])
print(v)
plt.plot(*v)
plt.show()

# Create an identity matrix (diagonal matrix of 1s and 0s)
identity_matix = np.array([[1,0],
                           [0,1]])

plt.plot(*np.dot(v, identity_matix))
plt.show()

# Create a matrix we can use to mutiply against our original matrix, v
multiplication_matrix = np.array([[2,0],
                                  [0,2]])

plt.plot(*multiplication_matrix)
plt.show()

# Find the Dot Procuct (Matrix mutiplication) of V and the multiplication matrix
plt.plot(*np.dot(v, multiplication_matrix))
plt.show()

