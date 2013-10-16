from copy import copy
import numpy as np
import matplotlib.pyplot as pp
from math import pi, exp,log

sigma = 0.1
beta  = 1.0 / pow(sigma,2) # this is the beta used in Bishop Eqn. 6.59
N_test = 100
x_test = np.linspace(-1,1,N_test) 
mu_test = np.zeros( N_test )

def true_mean_function( x ):
    return np.sin( 2*pi*(x+1) )

def add_noise( y, sigma ):
    return y + sigma*np.random.randn(len(y))

def generate_t( x, sigma = sigma):
    return add_noise( true_mean_function( x), sigma )


y_test = true_mean_function( x_test )
t_test = add_noise( y_test, sigma )

showExamplePlot = False
if showExamplePlot:
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

def plotjeh(theta):
    fig = pp.figure()
    for t in theta:
        K = computeK(x_test,x_test,t)
        y = np.random.multivariate_normal(np.zeros(len(x_test)),K,5)
        for i in xrange(5):
            pp.plot(x_test,y[i])
        distance = 2*(np.diag(K)**0.5)
        pp.fill_between(x_test, distance, - distance, alpha=0.1)
        pp.plot(x_test, np.zeros(x_test.shape), '--')
        pp.suptitle(str(t))
        pp.show()

def gp_predictive_distribution( x_train, t_train, x_test, theta, C = None ):
    if C == None:
        C = computeK(x_train, x_train, theta)
        # Add a 1\beta on the diagonal
        for i in xrange(C.shape[0]):
            C[i][i] += beta**-1
    mean = np.zeros(x_test.shape[0])
    variance = np.zeros(x_test.shape[0])
    for i in xrange(x_test.shape[0]):
        k = np.zeros(x_train.shape)
        c = k_n_m(x_test[i], x_test[i], theta) + beta**-1
        for j in xrange(x_train.shape[0]):
            k[j] = k_n_m(x_train[j], x_test[i], theta)
        mean[i] = k.dot(np.linalg.inv(C)).dot(t_train)
        variance[i] = c - k.dot(np.linalg.inv(C)).dot(k).T
    return mean, variance

def gp_log_likelihood(x_train, t_train, theta, C=None, invC=None):
    if C == None or invC == None:
        C = computeK(x_train, x_train, theta)
        # Add a 1\beta on the diagonal
        for i in xrange(C.shape[0]):
            C[i][i] += beta**-1
        invC = np.linalg.inv(C)
        
    log_likelihood = - 0.5 * (log(np.linalg.det(C)) + t_train.dot(invC).dot(t_train.T) + len(t_train) * log(2*pi))
    return log_likelihood

def gp_plot( x_test, y_test, mu_test, var_test, x_train, t_train, theta, beta, title ):
    # x_test: 
    # y_test:   the true function at x_test
    # mu_test:   predictive mean at x_test
    # var_test: predictive covariance at x_test 
    # t_train:  the training values
    # theta:    the kernel parameters
    # beta:      the precision (known)
    
    # the reason for the manipulation is to allow plots separating model and data stddevs.
    std_total = np.sqrt(var_test)         # includes all uncertainty, model and target noise 
    std_model = np.sqrt( std_total**2 - 1.0/beta ) # remove data noise to get model uncertainty in stddev
    std_combo = std_model + np.sqrt( 1.0/beta )    # add stddev (note: not the same as full)
    
    pp.plot( x_test, y_test, 'b', lw=3)
    pp.plot( x_test, mu_test, 'k--', lw=2 )
    pp.fill_between( x_test, mu_test+2*std_combo,mu_test-2*std_combo, color='k', alpha=0.25 )
    pp.fill_between( x_test, mu_test+2*std_model,mu_test-2*std_model, color='r', alpha=0.25 )
    pp.plot( x_train, t_train, 'ro', ms=10 )
    pp.suptitle(title)
    pp.show()


if __name__ == "__main__":
    theta = [[1.00,4.00,0.00,0.00],[9.00,4.00,0.00,0.00],[1.00,64.00,0.00,0.00],[1.00,0.25,0.00,0.00],[1.00,4.00,10.00,0.00],[1.00,4.00,0.00,5.00]]
    
    partOne = False
    partTwo = False
    partThree = True

    if partOne:
        plotjeh(theta)

    if partTwo:
        N_train = 50
        x_train = np.linspace(-1,1,N_train)
        t_train = generate_t(x_train)

        print gp_predictive_distribution(x_train, t_train, x_test, theta[0], None)
        print gp_log_likelihood(x_train, t_train, theta[0])

    if partThree:
        N_train = 2
        x_train = np.linspace(-1,1,N_train)
        t_train = generate_t(x_train)
        for t in theta:
            title = str(t) + " " + str(gp_log_likelihood(x_train, t_train, t))
            mu_test, var_test = gp_predictive_distribution( x_train, t_train, x_test, t)
            gp_plot(x_test, y_test, mu_test, var_test, x_train, t_train, t, beta, title)











    
