import numpy as np
import matplotlib.pyplot as plt 
import numpy.matlib 


a = np.empty([4,4]) #Empty Array
a = np.arange(10) #Full Array
a = np.full((4,4), 0) #Array filled with all 0
a = np.full((4,4), 1) #Array filled with all 1
np.array([1,1,1,1]) in a #Check if array contains a specified row
a = a[~np.isnan(a).any(axis=1)] #Remove rows in Numpy array that contains non-numeric values

#Remove single-dimensional entries from the shape of an array
a = np.array([[[0], [2], [4]]])  
a.shape
np.squeeze(a).shape

# Find the number of occurrences of a sequence in a NumPy array
occurrences = np.count_nonzero(a == 1) 

#Find the most frequent value in a NumPy array
a = np.full((4), [1, 3, 1, 4]) 
np.bincount(a).argmax()

# Combining a one and a two-dimensional NumPy Array
a = np.arange(5)  
b = np.arange(10).reshape(2,5)
for c, d in np.nditer([a, b]): 
    print("%d:%d" % (c, d),)
    
#How to build an array of all combinations of two NumPy arrays?
a = np.arange(10) 
b = np.full(4, 10)
c = np.array(np.meshgrid(a, b)).T.reshape(-1, 2) 

#How to add a border around a NumPy array?
a = np.full((2, 2), 1) 
b = np.pad(a, pad_width=1, mode='constant', constant_values=0) 

#How to compare two NumPy arrays?
a = np.full((4, 4), 1) 
b = np.full((4,4), 10)
comparison = a == b 
equal_arrays = comparison.all() 
    
#How to check whether specified values are present in NumPy array?
a = np.full((4, 4), 1) 
print(2 in a)

#How to get all 2D diagonals of a 3D NumPy array?
a = np.arange(3 * 4 * 4).reshape(3, 4, 4) 
b = np.diagonal(a, axis1 = 1, axis2 = 2) 

#Flatten a Matrix in Python using NumPy
a = np.full((2,2), 1) 
b = a.flatten() 

#Move axes of an array to new positions
a = np.zeros((3, 4, 5))
b = np.moveaxis(a, 0, -1).shape

#Interchange two axes of an array
a = np.array([[2, 4, 6]]) 
b = np.swapaxes(a, 0, 1) 

#NumPy – Fibonacci Series using Binet Formula
a = np.arange(1, 11) 
lengthA = len(a) 
r5 = np.sqrt(5) 
alpha = (1 + r5) / 2
beta = (1 - r5) / 2
Fn = np.rint(((alpha ** a) - (beta ** a)) / (r5)) 

#Counts the number of non-zero values in the array
a = np.full((4,4), 0)
b = np.count_nonzero(a) 

#Count the number of elements along a given axis
b = np.size(a, 0)

#Trim the leading and/or trailing zeros from a 1-D array
a = np.array((0, 0, 0, 0, 1, 5, 7, 0, 6, 2, 9, 0, 10, 0, 0)) 
b = np.trim_zeros(a)

#Change data type of given numpy array
a = a.astype('float64') 
print(a.dtype) 

#Reverse a numpy array
b = np.flipud(a) 

#How to make a NumPy array read-only?
a.flags.writeable = False

#Get the maximum value from given matrix
a = np.arange(1,10).reshape((3, 3))
b = np.amax(a)

#Get the minimum value from given matrix
b = np.amin(a)

#Find the number of rows and columns of a given matrix using NumPy
b = a.shape

#Select the elements from a given matrix
a = np.matrix('[1, 2, 3, 4; 3, 1, 5, 6]') 
b = np.choose([1, 0, 1, 0], a)

#Find the sum of values in a matrix]
a = np.arange(1,10).reshape((3, 3))
b = np.sum(a)

#Calculate the sum of the diagonal elements of a NumPy array
b = np.trace(a)

#Adding and Subtracting Matrices in Python
a = np.array([[1, 2], [3, 4]]) 
b = np.array([[4, 5], [6, 7]]) 
c = np.add(a, b)
c = np.subtract(a, b)

#Ways to add row/columns in numpy array
a = np.array([[1, 2, 3], [45, 4, 7], [9, 6, 10]])
b = np.array([1, 2, 3])
c = np.column_stack((a, b))

#Matrix Multiplication in NumPy
a = np.array([[1, 2], [3, 4]]) 
b = np.array([[4, 5], [6, 7]]) 
c = np.dot(a, b)

#Get the eigen values of a matrix
c = np.linalg.eig(a)

#How to Calculate the determinant of a matrix using NumPy?
c = np.linalg.det(a)

#How to inverse a matrix using NumPy
c = np.linalg.inv(a)

#How to count the frequency of unique values in NumPy array?
u, f = np.unique(a, return_counts = True) 

#Multiply matrices of complex numbers using NumPy in Python
a = np.array([2+3j, 4+5j]) 
b = np.array([8+7j, 5+6j]) 
c = np.vdot(a, b) 

#Compute the outer product of two given vectors using NumPy in Python
c = np.outer(a, b)

#Calculate inner, outer, and cross products of matrices and vectors using NumPy
c = np.inner(a, b)
d = np.outer(a, b)
e = np.cross(a, b)

#Compute the covariance matrix of two given NumPy arrays
c = np.cov(a, b)

#Compute the Kronecker product of two mulitdimension NumPy arrays
c = np.kron(a, b)

#Replace NumPy array elements that doesn’t satisfy the given condition
a = np.array([75.42436315, 42.48558583, 60.32924763]) 
a[a> 50.] = 15.50

#Return the indices of elements where the given condition is satisfied
b = np.where(a<4) 

#Replace NaN values with average of columns
a = np.array([[1.3, 2.5, 3.6, np.nan],  
                      [2.6, 3.3, np.nan, 5.5], 
                      [2.1, 3.2, 5.4, 6.5]])
b = np.nanmean(a, axis = 0) 
str(b)
c = np.where(np.isnan(a)) 
a[c] = np.take(b, c[1]) 
  
#Replace negative value with zero in numpy array
a = np.array([1, 2, -3, 4, -5, -6]) 
a[a<0] = 0

#How to get values of an NumPy array at certain index positions?
a = np.array([11, 10, 22, 30, 33]) 
b = np.array([1, 15, 60]) 
a.put([0, 4], b) 

#Find indices of elements equal to zero in a NumPy array
a = np.array([1, 0, 2, 0, 3, 0, 0, 5, 6, 7, 5, 0, 8]) 
res = np.where(a == 0)[0] 

#How to Remove columns in Numpy array that contains non-numeric values?
a = np.array([[10.5, 22.5, np.nan], [41, 52.5, np.nan]]) 
a[:, ~np.isnan(a).any(axis=0)]

#How to access different rows of a multidimensional NumPy array?
a = np.array([[10, 20, 30], [40, 5, 66], [70, 88, 94]])
b = a[[0,2]] 

#Get row numbers of NumPy array having element larger than X
a = np.array([[1, 2, 3, 4, 5], [10, -3, 30, 4, 5], [3, 2, 5, -4, 5], [9, 7, 3, 6, 5]]) 
x = 6  
b  = np.where(np.any(b > x, axis = 1)) 

#Get filled the diagonals of NumPy array
a = np.array([[1, 2], [2, 1]]) 
np.fill_diagonal(a, 5)

#Check elements present in the NumPy array
a = np.array([[2, 3, 0],[4, 1, 6]])
print(2 in a) 

#Find a matrix or vector norm using NumPy
a = np.arange(10) 
b = np.linalg.norm(a) 

#Calculate the QR decomposition of a given matrix using NumPy
a = np.array([[1, 2, 3], [3, 4, 5]]) 
q, r = np.linalg.qr(a) 

#Compute the condition number of a given matrix using NumPy
b =  np.linalg.cond(a) 

#Calculate the Euclidean distance using NumPy
a = np.array((1, 2, 3)) 
b = np.array((1, 1, 1))  
c = np.linalg.norm(a - b) 

#Create a Numpy array with random values
a = np.random.rand(5) 

#How to choose elements from the list with different probability using NumPy?
a = [10, 20, 30, 40, 50]  
b = np.random.choice(a) 

#Generate Random Numbers From The Uniform Distribution using NumPy
a = np.random.uniform(size=4) 

#Get Random Elements form geometric distribution
a = np.random.geometric(0.65, 1000) 

#Get Random elements from Laplace distribution
a = np.random.laplace(1.45, 15, 1000) 
count, bins, ignored = plt.hist(a, 30, density = True) 
plt.show()

#Return a Matrix of random values from a uniform distribution
a = np.random.uniform(-5, 5, 5000) 
plt.hist(a, bins = 50, density = True) 
plt.show()

#Return a Matrix of random values from a Gaussian distribution
a = np.matlib.randn((3, 4))  

#How to get the indices of the sorted array using NumPy in Python?
a = np.array([10, 52, 62, 16, 16, 54, 453]) 
b = np.argsort(a) 

#Finding the k smallest values of a NumPy array
a = np.array([23, 12, 1, 3, 4, 5, 6]) 
k = 5  
b = np.sort(a) 
print(b[:k])

#How to get the n-largest values of an array using NumPy?
a = np.array([2, 0,  1, 5, 4, 1, 9])
b = np.argsort(a) 
n = 1
a = b[-n : ]

#Sort the values in a matrix
a = np.matrix('[4, 1; 12, 3]') 
a.sort() 

#Find the indices into a sorted array
i = np.argsort(a) 














