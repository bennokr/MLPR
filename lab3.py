# -*- coding: utf-8 -*-
"""
Spyder Editor

This temporary script file is located here:
c:\WINNT\profiles\10450793\.spyder2\.temp.py
"""
import numpy as np
import matplotlib.pyplot as pp
from math import pi,exp,log

sigma = 0.1
beta  = 1.0 / pow(sigma,2) # this is the beta used in Bishop Eqn. 6.59
N_test = 100
x_test = np.linspace(-1,1,N_test); 
mu_test = np.zeros( N_test )
def true_mean_function( x ):
    return np.sin( 2*pi*(x+1) )

def add_noise( y, sigma ):
    return y + sigma*np.random.randn(len(y))

def generate_t( x, sigma ):
    return add_noise( true_mean_function( x), sigma )
y_test = true_mean_function( x_test )
t_test = add_noise( y_test, sigma )
pp.plot( x_test, y_test, 'b-', lw=2)
pp.plot( x_test, t_test, 'go')

def k_n_m( xn, xm, thetas): # xn and xm are scalers
    kernel = thetas[0]*exp((-thetas[1]/2)*((xn-xm)**2)) + thetas[2] + thetas[3]*xn*xm
    return kernel

def computeK(X1, X2, thetas):
    matrix = np.zeros((len(X1),len(X2)))
    for x in xrange(len(X1)):
        for x2 in xrange(len(X2)):
            matrix[x][x2] = k_n_m( X1[x], X2[x2], thetas)
    return matrix

def plotjeh (theta):
    fig = pp.figure()
    for t in theta:
        y= np.random.multivariate_normal(np.zeros(len(x_test)),computeK(x_test,x_test,t),5)
        for i in xrange(5):
            pp.plot(x_test,y[i])
        fig.suptitle(str(t))
        pp.show()

if __name__ == "__main__":
    theta = [[1.00,4.00,0.00,0.00],[9.00,4.00,0.00,0.00],[1.00,64.00,0.00,0.00],[1.00,0.25,0.00,0.00],[1.00,4.00,10.00,0.00],[1.00,4.00,0.00,5.00]]
    plotjeh(theta)
    

