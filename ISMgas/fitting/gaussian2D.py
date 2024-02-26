import numpy as np
from astropy.modeling import models, fitting
from astropy.io import fits 
import matplotlib.pyplot as plt

class gaussian2D:
    def __init__(self, data):
        self.data     = data
        self.results  = {}
    
    def _fit(self, kwargs):
        # initialize a fitting routine
        fit = fitting.LevMarLSQFitter()

        # initialize a linear model
        g2d_init = models.Gaussian2D(**kwargs['gaussian2D'])

        # fit the data with the fitter, and plot the result
        x,y = np.mgrid[:(np.shape(self.data))[0], :(np.shape(self.data))[1]]
        g2d_fitted = fit(g2d_init, x, y,  self.data)
        
        self.results = {
            'data' : self.data,
            'fit'  : g2d_fitted(x,y),
            'residual' : self.data  - g2d_fitted(x,y),
            'c_x'      : g2d_fitted.x_mean.value,
            'c_y'      : g2d_fitted.y_mean.value,
            'A'        : g2d_fitted.amplitude.value,
            'theta'    : g2d_fitted.theta.value,
            'stddev_x' : g2d_fitted.x_stddev.value,
            'stddev_y' : g2d_fitted.y_stddev.value,
        }
    
        return(self.results)
    
    def _plotResults(self,results):
        plt.figure(figsize=(12,3), dpi = 200)

        plt.subplot(1,3,1)
        plt.imshow(results['data'], origin='lower')
        plt.scatter(results['c_y'], results['c_x'], color = 'orange', marker= 'x', s=100)
        plt.colorbar()
        plt.title("Data")

        plt.subplot(1,3,2)
        plt.imshow(results['fit'], origin='lower')
        plt.colorbar()
        plt.title("Fit")

        plt.subplot(1,3,3)
        plt.imshow(results['residual'], origin='lower')
        plt.colorbar()
        plt.title("Data - Fit")
        