import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def RetData(dbname):
	data = open(dbname)
	data = data.read().splitlines()

	splitted_data = []

	for d in data:
		splitted_data.append(d.split(','))

	splitted_data = np.array(splitted_data)
	splitted_data = splitted_data.astype(float)

	return splitted_data


def plot(data, labels, w):
	fig, ax = plt.subplots()

	c0 = data[labels == 0]
	c1 = data[labels == 1]

	ax.scatter(c0[:,0], c0[:,1], c='red')
	ax.scatter(c1[:,0], c1[:,1], c='blue')

	a, b, c = w
	m = -a / b
	b = -c / b

	x = np.arange(np.min(data[:,0]), np.max(data[:,0]), 0.1)
	y = m * x + b
	plt.plot(x, y)

	plt.show()


def sigmoid(z):
	return 1 / ( 1 + np.exp(-z) )


def Logistic_Regression_via_GD(P,y,lr):
	ones_column = np.ones((P.shape[0], 1))
	P = np.hstack((ones_column, P))

	w = np.zeros(P.shape[1])
	w[0] = 1

	grad_test = 1

	Ptrans = P.transpose()

	while grad_test > 0.01:
		sigm = sigmoid(np.matmul(w, Ptrans))
		grad = np.matmul(Ptrans, sigm-y)
		w = w - ((lr * grad) / P.shape[0])
		grad_test = np.linalg.norm(grad)
	return w[1:], w[0]


def predict(w,b,p):
	multiplied = np.dot(w, p)
	multiplied = multiplied + b
	sigmoid_result = sigmoid(multiplied)
	rounded_result = np.rint(sigmoid_result)
	return rounded_result

accuracy = 0
current_accuracy = 0
maximum_w = 0
maximum_b = 0
maximum_acc = 0
for i in range(20):
	data = RetData('exams.csv')

	train, test = train_test_split(data, test_size=0.1)

	train, label_train = train[:, :2], train[:, 2]

	test, label_tests = test[:, :2], test[:, 2]

	scaler = StandardScaler()

	train = scaler.fit_transform(train)
	test = scaler.fit_transform(test)
	lr = 0.004
	correct_identified = 0
	w, b = Logistic_Regression_via_GD(train, label_train, lr)
	for label, testa in zip(label_tests, test):
		c = predict(w, b, testa)
		if c == label:
			correct_identified = correct_identified + 1
	#print("Current Accuracy: " , accuracy)
	current_accuracy = correct_identified / label_tests.size
	if(maximum_acc < current_accuracy):
		maximum_b = b
		maximum_w = w
		maximum_acc = current_accuracy
	accuracy = accuracy + correct_identified / label_tests.size
print(f"Avg test accuracy: {accuracy / 20 * 100}%")
