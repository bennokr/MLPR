from copy import copy
import numpy as np
import matplotlib.pyplot as plt
from math import pi, sin


def gen_sinusoidal(N):
	"""
	Draws N samples from a sinus with consistent intervals
	"""
	x = np.linspace(0, 2*pi, N)
	t = np.random.normal(0,0.2,N) + np.array(map(sin, x))
	return x, t

def calculate_phi(x,M):
	"""
	Calculates matrix phi for the samples x
	"""
	phi = np.tile(copy(x),(M,1))
	for i in xrange(len(phi)):
		phi[i] = phi[i]**i
	return np.matrix(phi.T)

def fit_polynomial(x, t, M):
	"""
	Gives best-fit parameters for an M degree polinomial,
	x is input an t is target
	Works according to page XXX in the book XXX
	"""
	# Calculate phi
	phi = calculate_phi(x, M)

	# Calculate best fit (and do a type cast)
	Wml =  phi.T.dot(phi).I.dot(phi.T).dot(t)
	return np.array(Wml)[0]

def fit_polynomial_reg(x,t,M,lamb):
	# Calculate phi
	phi = calculate_phi(x, M)

	# Calculate best fit (and do a type cast)
	Wml = (lamb * np.identity(M) + phi.T.dot(phi)).I.dot(phi.T).dot(t)
	return np.array(Wml)[0]


def polinomialValue(w, x):
	"""
	Calculates the value of polinomial w at point x
	"""
	return sum( [w[i] * x**i for i in xrange(len(w)) ] )



if __name__ == "__main__":
	"""
	Calculations
	"""
	## Generate empirical data
	x, t = gen_sinusoidal(9)

	## Generate polinomial models for empirical data
	polinomialDegrees = [1,3,9]

	#models = map(lambda M : fit_polynomial(x,t,M), polinomialDegrees)     			# For 1.3
	models = map(lambda M : fit_polynomial_reg(x, t, M, 0.2), polinomialDegrees)	# For 1.4

	"""
	Plotting
	"""
	## Set axis
	plotOffset = 0.1
	plt.axis([0-plotOffset, 2*pi+plotOffset, -1-plotOffset, 1+plotOffset])

	## Plot the empirical points
	plt.plot(x, t, 'o')

	## Plot the 'true function'
	x = np.linspace(0, 2*np.pi, 100)
	plt.plot(x, np.sin(x), lw=3)

	## Plot the various models
	for i in xrange(len(polinomialDegrees)):
		plt.plot(x, map(lambda z : polinomialValue(models[i], z), x), label="P "+str(polinomialDegrees[i]))
	## Save the figure to disk
	#plt.savefig('1.3.png')

	plt.show()


