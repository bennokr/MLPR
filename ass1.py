import numpy as np
from scipy import misc
import pylab as plt
import matplotlib.pyplot as plt
from math import pi, sin

def gen_sinusoidal(N):
	x = np.linspace(0, 2*pi, N)
	t = np.random.normal(0,0.2,N) + np.array(map(sin, x))
	return x, t

# x, t = gen_sinusoidal(100)
# plt.plot(x, t, 'o')
# plt.ylabel('some numbers')
# plt.show()

def fit_polynomial(x, t, M):
	pass
