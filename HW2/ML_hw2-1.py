import numpy as np
import os
import struct
import matplotlib.pyplot as plt
from math import log, sqrt, exp, pi, factorial
import argparse


mode = int(input("mode:"))

def load_mnist(path, kind='train'):
	labels_path = os.path.join(path,'%s-labels.idx1-ubyte' % kind)
	images_path = os.path.join(path,'%s-images.idx3-ubyte' % kind)

	with open(labels_path, 'rb') as lbpath:
		magic, n = struct.unpack('>II', lbpath.read(8))
		labels = np.fromfile(lbpath, dtype=np.uint8)


	with open(images_path, 'rb') as imgpath:
		magic, n, rows, cols = struct.unpack('>IIII', imgpath.read(16))
		images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

	return images, labels, n



###########################
#In Discrete mode
###########################
def DiscreteMode():
	train_img, train_label, train_num = load_mnist("./img/", kind='train')
	test_img, test_label, test_num = load_mnist("./img/", kind='t10k')
	train_img = train_img.reshape(train_num, 28, 28)
	test_img = test_img.reshape(test_num, 28, 28)

	pixelCount = np.ones([10, 28*28, 32])
	classCount = np.zeros(10)

	if (os.path.isfile('./pixelCount.npy') and os.path.isfile('./classCount.npy')):
		pixelCount = np.load('./pixelCount.npy')
		classCount = np.load('./classCount.npy')
	else:
		for idx, data in enumerate(train_img):
			classCount[train_label[idx]] += 1
			for pixel in range(28*28):
				pixelCount[train_label[idx], pixel, (data[pixel//28, pixel%28] >> 3)] += 1
		np.save('./pixelCount.npy', pixelCount)
		np.save('./classCount.npy', classCount)


	error = 0
	Posterior = np.zeros([test_num, 10])
	if os.path.isfile('./Posterior_D.npy'):
		Posterior = np.load('./Posterior_D.npy')
	else:
		for idx, data in enumerate(test_img):
			for num in range(10):
				for pixel in range(28*28):
					Posterior[idx][num] += log(pixelCount[num][pixel][data[pixel//28, pixel%28]>>3]/classCount[num])

				Posterior[idx][num] += log(classCount[num]/60000)
			Posterior[idx] = Posterior[idx] / sum(Posterior[idx])   
		np.save('./Posterior_D.npy', Posterior) 

	for idx, data in enumerate(test_img): 
		predict = np.argmin(Posterior[idx])
		answer = test_label[idx]
		if predict != answer:
			error += 1

	#print result of first image
	print('Posterior (in log scale):')
	for i in range(10):
		print('{}: {}'.format(i, Posterior[0][i]))
	print('Prediction: {}, Ans: {}\n'.format(np.argmin(Posterior[0]), test_label[0]))

	#print result of second image
	print('Posterior (in log scale):')
	for i in range(10):
		print('{}: {}'.format(i, Posterior[1][i]))
	print('Prediction: {}, Ans: {}\n'.format(np.argmin(Posterior[1]), test_label[1]))

	print('\n.. all other predictions goes here ...\n')
	print('\nImagination of numbers in Bayesian classifier:\n')


	for i in range(10):
		print('\n{}:'.format(i))
		for pixel in range(28*28):
			res = np.argmax(pixelCount[i, pixel, :])
			if res <= 15:
				print("0", end = ' ')
			else:
				print("1", end = ' ')
			if pixel % 28 == 27:
				print('')

	print('Error rate: {}'.format(error/test_num))


###########################
#In Continuous mode
###########################
def Gaussian(u, v):
	return lambda x : (exp(-1*(((x - u)**2) / (2*v) ) ) / sqrt(2*pi*v))

def Continuous():
	train_img, train_label, train_num = load_mnist("./img/", kind='train')
	test_img, test_label, test_num = load_mnist("./img/", kind='t10k')
	train_img = train_img.reshape(train_num, 28, 28)
	test_img = test_img.reshape(test_num, 28, 28)


	classCount = np.zeros(10)
	if os.path.isfile('./classCount.npy'):
		classCount = np.load('./classCount.npy')
	else:
		for idx, data in enumerate(train_img):
			classCount[train_label[idx]] += 1

	pixelGaussian = np.zeros([10, 28*28, 2])
	if os.path.isfile('./pixelGaussian.npy'):
		pixelGaussian = np.load('./pixelGaussian.npy')
	else:
		pixelDatas = [[[] for _ in range(28*28)] for _ in range(10)]
		for idx, data in enumerate(train_img):
			for pixel in range(28*28):
				pixelDatas[train_label[idx]][pixel].append(data[pixel//28, pixel%28])

		Variance_zero = 1000
		count = 0
		for i in range(10):
			for j in range(28*28):
				u = sum(pixelDatas[i][j]) / classCount[i]
				v = sum([abs(x - u)**2 for x in pixelDatas[i][j]]) / classCount[i]
			
				pixelGaussian[i][j][0] = u
				if v <= Variance_zero:
					count += 1
					v = Variance_zero
				pixelGaussian[i][j][1] = v
		np.save('./pixelGaussian.npy', pixelGaussian)


	error = 0
	Posterior = np.zeros([test_num, 10])
	if os.path.isfile('./Posterior_G.npy'):
		Posterior = np.load('./Posterior_G.npy')
	else:
		for idx, data in enumerate(test_img):
			for num in range(10):
				for pixel in range(28*28):
					u = pixelGaussian[num][pixel][0]
					v = pixelGaussian[num][pixel][1]
					Posterior[idx][num] += log(Gaussian(u,v)(data[pixel//28, pixel%28]))

				Posterior[idx][num] += log(classCount[num]/60000)

			Posterior[idx] = Posterior[idx] / sum(Posterior[idx])
			if idx%100 == 0:
				print(idx)
		np.save('./Posterior_G.npy', Posterior)

	for idx, data in enumerate(test_img): 
		predict = np.argmin(Posterior[idx])
		answer = test_label[idx]
		if predict != answer:
			error += 1

	#print result of first image
	print('Posterior (in log scale):')
	for i in range(10):
		print('{}: {}'.format(i, Posterior[0][i]))
	print('Prediction: {}, Ans: {}\n'.format(np.argmin(Posterior[0]), test_label[0]))

	#print result of second image
	print('Posterior (in log scale):')
	for i in range(10):
		print('{}: {}'.format(i, Posterior[1][i]))
	print('Prediction: {}, Ans: {}\n'.format(np.argmin(Posterior[1]), test_label[1]))

	print('\n.. all other predictions goes here ...\n')
	print('\nImagination of numbers in Bayesian classifier:\n')

	for i in range(10):
		print('\n{}:'.format(i))
		for pixel in range(28*28):
			res = pixelGaussian[i, pixel, 0]
			if res <= 127:
				print("0", end = ' ')
			else:
				print("1", end = ' ')
			if pixel % 28 == 27:
				print('')

	print('Error rate: {}'.format(error/test_num))

if mode == 0:
	DiscreteMode()
else:
	Continuous()