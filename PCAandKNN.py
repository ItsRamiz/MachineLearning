import sys
import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import cdist


def RetData(name):
	data = pd.read_csv(name) #Retrieiving the CSV file
	data = np.array(data) #Turning it into a two-dimensional array
	labels = data[:,0]	#Extracting all the elements from the first column
	data = data[:,1:] #Extracting all the elemenets from the second column onwards
	return labels,data

def plot_cdf(data):
	sorted_data = np.sort(data)[::-1]

	data_cumsum = np.cumsum(sorted_data)
	data_normalized = data_cumsum / data_cumsum[-1]

	# Plot the CDF of eigenvalues
	plt.plot(np.arange(1, len(sorted_data)+1), data_normalized)
	plt.xlabel('Principal Component')
	plt.ylabel('Cumulative Proportion of Variance')
	plt.title('Cumulative Distribution Function of Eigenvalues')
	plt.show()

def PCA(data,k):
	#print(data.shape)
	normalized_data = data - data.mean() #Normalizing the data
	# moving to mean (0,0)
	S = np.matmul(normalized_data.transpose(), normalized_data)
	#print("Hi")
	#print(S.shape)
	evalues, evectors = np.linalg.eig(S)  # the vectors are the columns
	sorted_evalues = evalues.argsort()[::-1]  # indexes of k largest eig
	E = evectors[:, sorted_evalues][:, :k]
	#print("E:")
	#print(E.shape)
	#print("S: ")
	#print(normalized_data.shape)
	y = np.matmul(E.transpose(), normalized_data.transpose())
	y = y.transpose()
	#print(y.transpose().shape)
	return y, E.transpose()

def PCA_test(data,mean,E):
	normalized_pics = data - mean
	y = np.matmul(E,normalized_pics.transpose())
	y = y.transpose()
	return y


def kNN_classify(data,labels,test,k):
	img_dists = cdist(data, test).transpose()
	img_dists = img_dists.argsort()
	smallestK = img_dists[:,:k]
	classified = np.array([np.bincount(labels[ind]).argmax() for ind in smallestK])
	return classified


accuracy = 0

label_train, data = RetData("fashion-mnist_train.csv")
label_test, test = RetData("fashion-mnist_test.csv")
train,eig = PCA(data,81)
test = PCA_test(test,data.mean(),eig)
k = 6
correct_identified = 0
images = np.array(test)
c = kNN_classify(train,label_train,images,k)
correct_identified = label_test == c
accuracy = correct_identified / len(label_test)
accuracy = np.sum(accuracy)
print(f"Test accuracy is: {accuracy * 100}%")

