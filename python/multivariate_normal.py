import numpy as np
import numpy.matlib as nm
from svgd import SVGD
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

class MVN:
    def __init__(self, mu, A):
        self.mu = mu
        self.A = A
    
    def dlnprob(self, theta):
        # return -1*np.matmul(theta-nm.repmat(self.mu, theta.shape[0], 1), self.A)
        return -(np.matmul(np.linalg.inv(self.A), (theta-nm.repmat(self.mu, theta.shape[0], 1)).T )).T
    
if __name__ == '__main__':
    A = np.array([[0.2260,0.1652],[0.1652,0.6779]])
    mu = np.array([-0.6871,0.8010])
    
    model = MVN(mu, A)

    x, y = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    pos = np.dstack((x, y))

    # Create the 2D normal distribution
    rv = multivariate_normal(mu, A)

    # Plot the probability density function
    plt.contourf(x, y, rv.pdf(pos), cmap='viridis')
    plt.title('2D Normal Distribution')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.colorbar(label='Probability Density')


    x0 = np.random.normal(0,1, [50,2]);

    plt.scatter(x0[:, 0], x0[:, 1], color='red', label='Random Points', alpha=0.7)
    plt.show()

    print("ground truth: ", mu)

    theta = SVGD().update(x0, model.dlnprob, n_iter=1000, stepsize=0.01)

    plt.contourf(x, y, rv.pdf(pos), cmap='viridis')
    plt.title('2D Normal Distribution')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.colorbar(label='Probability Density')

    plt.scatter(theta[:, 0], theta[:, 1], color='red', label='Random Points', alpha=0.7)
    plt.show()
    
    # print("ground truth: ", mu)
    print("svgd: ", np.cov(theta.T))
    print("ground truth: ", A)
