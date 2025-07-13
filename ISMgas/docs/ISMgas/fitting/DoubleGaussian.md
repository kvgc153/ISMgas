Module ISMgas.fitting.DoubleGaussian
====================================

Classes
-------

`DoubleGaussian(x, y, yerr, inst_sigma=1)`
:   Class which performs the double / single gaussian fit to the data.
    Assumes that continuum is at a constant level. Preprocess data if needed   
    
    x, y, yerr : required 
    inst_sigma : (optional) minimum sigma that the spectra can resolve. Essential for derving velocity measurements

    ### Static methods

    `fitGaussian(specDict, seedsInit=12345, nseeds=5, qtys=[['derived_results', 'v50'], ['derived_results', 'v25'], ['derived_results', 'v05'], ['derived_results', 'delta_v90'], ['derived_results', 'EW_kms']], double_gaussian=True, constraint_option=1, priors=[], plotting=True, inst_sigma=1)`
    :   A convenience function to quantify the uncertainity in the DG/SG fits.
        This fits the absorption profile using multiple seeds and computes
        the median and MAD values for all the quantities (e.g., v50, deltav90).

    ### Methods

    `derivedVelocityMeasurements(self, A_out, A1_out, v_out, v1_out, sig_out, sig1_out, double_gaussian, outflow=False)`
    :   This function returns all the derived velocity measurements given the double gaussian coefficients
        
        - v01
        - v05
        - v10
        - v25
        - v50
        - v75
        - v90
        - v95
        - v99,
        - $\Delta$ v98
        - $\Delta$ v90
        - $\Delta$ v80
        - $\Delta$ v50
        
        #### Example
        
        ```
        obj.derivedVelocityMeasurements(1,2,3,4,5,6,double_gaussian=True,outflow=False)
        
        {'v01': -10,
         'v05': -6,
         'v10': -4,
         'v25': -1,
         'v50': 3,
         'v75': 7,
         'v90': 11,
         'v95': 13,
         'v99': 17,
         'delta_v98': 27,
         'delta_v90': 19,
         'delta_v80': 15,
         'delta_v50': 8}
        
        ```

    `find_nearest(self, input_array, value)`
    :   used in derivedVelocityMeasurements. 
        Given an array and a value X, returns the value in array nearest to X.

    `fitting(self, niter=100, seed=12345, constraint_option=1, double_gaussian=True, verbose=False, continuum=[], stepsize=0.005)`
    :   Given x, y and ysigma the function returns the parameters of a
        double gaussian fit of the form :
        1- (A*(exp(-(w**2)/2.))-  A1*(exp(-(w1**2)/2.))
        
        or a single gaussian fit of the form
        1- (A*(exp(-(w**2)/2.))
        
        
        #### Example
        ```
        A = 0.8
        A1 = 0.6
        sig = 220
        sig1 = 150
        v = -250
        v1 = 10
        
        
        obj.combined_wavelength = np.arange(-2000,2000,25)
        obj.combined_flux = 1-(
            A*np.exp(-(v-obj.combined_wavelength)**2/(sig**2)*2.)
            +A1*np.exp(-(v1-obj.combined_wavelength)**2/(sig1**2)*2.))
        
        
        plt.figure(figsize=(12,8))
        plt.plot(obj.combined_wavelength,obj.combined_flux )
        
        
        obj.combined_fluxerr = 0.05*np.ones(len(obj.combined_wavelength))
        results = obj.fitting(seed=12345,double_gaussian=True,verbose=False)
        
        plt.plot(results['fitted_wav'],results['fitted_flux'])
        
        ```

`plotFittingResults(**kwargs)`
:   

    ### Methods

    `plotChosenLines(self, **kwargs)`
    :

    `plotResiduals(self, **kwargs)`
    :   kwargs:
        - xlim1 :
        - ylim1 :
        - xticks1 :
        - yticks1 :
        - xtickshist :

    `plotVelocityFlux(self, **kwargs)`
    :   kwargs:
        - vline:
        - xlim:
        - ylim:
        - xticks:
        - yticks:
        - fontsize:
        - vline: Default is it plots v25_1

`unWrapFittingResults(seeds, double_gaussian, folderName, objid)`
:   This class is used to combine the different results from the DoubleGaussian class

    ### Methods

    `computeResults(self)`
    :

    `dataloader(self)`
    :

    `saveResults(self)`
    :