import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from symfit import variables, parameters, Model, Fit, exp, GreaterThan, LessThan
from symfit.core.minimizers import BasinHopping

import time
from ISMgas.fitting.init_fitting import *
from ISMgas.SupportingFunctions import load_pickle, save_as_pickle, plotHistogram
from ISMgas.globalVars import *

class DoubleGaussian:
    '''
    Class which performs the double / single gaussian fit to the data.
    Assumes that continuum is at a constant level. Preprocess data if needed   
    '''
    def __init__(self, x, y, yerr, inst_sigma=1):
        '''
        x, y, yerr : required 
        inst_sigma : (optional) minimum sigma that the spectra can resolve. Essential for derving velocity measurements
        '''
       
        self.x            = x
        self.y            = y
        self.yerr         = yerr
        self.inst_sigma   = inst_sigma
        self.init_values  = {}
        self.results_dict = {}
        
        # use priors in init_fitting.py or make your own priors on the go!
        self.priors   = priorsInit()


    def find_nearest(self, input_array, value):
        '''
        used in derivedVelocityMeasurements. 
        Given an array and a value X, returns the value in array nearest to X.
        '''
        array   = np.asarray(input_array)
        idx     = (np.abs(array - value)).argmin()
        return array[idx]


    def derivedVelocityMeasurements(self, A_out, A1_out, v_out, v1_out, sig_out, sig1_out, double_gaussian, outflow=False):
        '''
            This function returns all the derived velocity measurements given the double gaussian coefficients

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

        '''
        v = np.arange(-1300,1300,1)
        if(outflow):
            v = np.arange(-1300,0,1)

        if(double_gaussian):
            ## NOTE: This uses (A*(exp(-(w**2)/2.))+  A1*(exp(-(w1**2)/2.)) whereas the fitting uses 1-[..]

            flux = np.array([ A_out*np.exp(-0.5*(v_out -i)**2/sig_out**2) +A1_out*np.exp(-0.5*(v1_out - i)**2/sig1_out**2) for i in v])

        elif(double_gaussian==False):

            flux = np.array([ A_out*np.exp(-0.5*(v_out -i)**2/sig_out**2)  for i in v])

        cdf = np.cumsum(flux)/np.max(np.cumsum(flux)) 

        ## Alternate implementation could be using np.interp but since we already know the functional form this is not necessary
        ## Due to the legacy reasons v05 here represents 95th percentile of absorption and v25 the 75th percentile of absorption 
        ## defined in the conventional sense. 
               
        measurements = {
            'v01': v[np.where(cdf==self.find_nearest(cdf,0.01))][0],
            'v05': v[np.where(cdf==self.find_nearest(cdf,0.05))][0],
            'v10': v[np.where(cdf==self.find_nearest(cdf,0.10))][0],
            'v25': v[np.where(cdf==self.find_nearest(cdf,0.25))][0],
            'v50': v[np.where(cdf==self.find_nearest(cdf,0.50))][0],
            'v75': v[np.where(cdf==self.find_nearest(cdf,0.75))][0],
            'v90': v[np.where(cdf==self.find_nearest(cdf,0.90))][0],
            'v95': v[np.where(cdf==self.find_nearest(cdf,0.95))][0],
            'v99': v[np.where(cdf==self.find_nearest(cdf,0.99))][0]
        }

        measurements['delta_v98']   = measurements['v99'] -  measurements['v01']
        measurements['delta_v90']   = measurements['v95'] -  measurements['v05']
        measurements['delta_v80']   = measurements['v90'] -  measurements['v10']
        measurements['delta_v50']   = measurements['v75'] -  measurements['v25']
        
        measurements['EW_kms']      = np.sum(flux) ## EW_kms = \sum C_f(v) * 1 (km/s) -- we are sampling it at 1 km/s

        return(measurements)


    def fitting(self,
                niter               = 100,
                seed                = 12345,
                constraint_option   = 1,
                double_gaussian     = True,
                verbose             = False,
                continuum           = [],
                stepsize            = 0.005
        ):
        '''
        Given x, y and ysigma the function returns the parameters of a
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

        '''

        wav_copy            = np.array(self.x.copy())
        final_profile_copy  = np.array(self.y.copy())
        final_error_copy    = np.array(self.yerr.copy())

        chi_sq_continuum = 0.
        if(len(continuum)>0):
            for i in continuum:
                continuum_profile   = final_profile_copy[(wav_copy>i[0]) & (wav_copy<i[1])]
                continuum_error     = final_error_copy[(wav_copy>i[0]) & (wav_copy<i[1])]
                chi_sq_continuum    += np.sum(((continuum_profile-1)**2)/continuum_error**2,axis=0)


        ##############################################
        ######### Priors and initialization ##########
        ##############################################

        x, y                      = variables('x,y')                      # x- velocity , y - flux
        v, sig, v1, sig1, A, A1   = parameters('v, sig, v1, sig1, A, A1') # parameters to optimize
        w                         = (v-x)/sig
        w1                        = (v1-x)/sig1
        
        ## Pick a random every single time the fitting is run ##
        np.random.seed(int(time.time())+seed)

        xdata = wav_copy

        ## At each point, generate a perturbed realization to fit by adding 1sigma noise
        ydata = final_profile_copy + np.array([np.random.uniform(low=-i,high=i) for i in final_error_copy])

        if(double_gaussian==True):
            
            A.value                     = np.random.uniform(low=self.priors[str(constraint_option)]["Amin"],high=self.priors[str(constraint_option)]["Amax"])
            self.init_values['A']       = A.value
            A.min                       = self.priors[str(constraint_option)]["Amin"]
            A.max                       = self.priors[str(constraint_option)]["Amax"]

            v.value                     = np.random.uniform(low=self.priors[str(constraint_option)]["vmin"],high=self.priors[str(constraint_option)]["vmax"])
            self.init_values['v']       = v.value
            v.min                       = self.priors[str(constraint_option)]["vmin"]
            v.max                       = self.priors[str(constraint_option)]["vmax"]

            sig.value                   = np.random.uniform(low=self.priors[str(constraint_option)]["sigmin"],high=self.priors[str(constraint_option)]["sigmax"])
            self.init_values['sig']     = sig.value
            sig.min                     = self.priors[str(constraint_option)]["sigmin"]
            sig.max                     = self.priors[str(constraint_option)]["sigmax"]

            A1.value                    = np.random.uniform(low=self.priors[str(constraint_option)]["A1min"],high=self.priors[str(constraint_option)]["A1max"])
            self.init_values['A1']      = A1.value
            A1.min                      = self.priors[str(constraint_option)]["A1min"]
            A1.max                      = self.priors[str(constraint_option)]["A1max"]

            v1.value                    = np.random.uniform(low=self.priors[str(constraint_option)]["v1min"],high=self.priors[str(constraint_option)]["v1max"])
            self.init_values['v1']      = v1.value
            v1.min                      = self.priors[str(constraint_option)]["v1min"]
            v1.max                      = self.priors[str(constraint_option)]["v1max"]

            sig1.value                  = np.random.uniform(low=self.priors[str(constraint_option)]["sig1min"],high=self.priors[str(constraint_option)]["sig1max"])
            self.init_values['sig1']    = sig1.value
            sig1.min                    = self.priors[str(constraint_option)]["sig1min"]
            sig1.max                    = self.priors[str(constraint_option)]["sig1max"]           
            
            model_dict = {
                y: self.priors[str(constraint_option)]["cont_lvl"] - A*(exp(-(w**2)/2.))-  A1*(exp(-(w1**2)/2.)).as_expr()
            }

            model = Model(model_dict)
            if(verbose):
                print(model)

            constraints = [
                LessThan(v-v1,self.priors[str(constraint_option)]["v-v1_min"]),
                GreaterThan(v-v1,self.priors[str(constraint_option)]["v-v1_max"])
            ]

            # Perform the fit - sigma_y is needed for chi^2 optimization
            fit = Fit(
                model,
                x             = xdata,
                y             = ydata,
                sigma_y       = final_error_copy,
                minimizer     = BasinHopping,
                constraints   = constraints
            )

        elif(double_gaussian==False):
            
            A.value                     = np.random.uniform(low=self.priors[str(constraint_option)]["Amin_SG"],high=self.priors[str(constraint_option)]["Amax_SG"])
            self.init_values['A']       = A.value
            self.init_values['A1']      = 0
            A.min                       = self.priors[str(constraint_option)]["Amin_SG"]
            A.max                       = self.priors[str(constraint_option)]["Amax_SG"]


            v.value                     = np.random.uniform(low=self.priors[str(constraint_option)]["vmin_SG"],high=self.priors[str(constraint_option)]["vmax_SG"])
            self.init_values['v']       = v.value
            self.init_values['v1']      = 0
            v.min                       = self.priors[str(constraint_option)]["vmin_SG"]
            v.max                       = self.priors[str(constraint_option)]["vmax_SG"]


            sig.value                   = np.random.uniform(low=self.priors[str(constraint_option)]["sigmin_SG"],high=self.priors[str(constraint_option)]["sigmax_SG"])
            self.init_values['sig']     = sig.value
            self.init_values['sig1']    = 0
            sig.min                     = self.priors[str(constraint_option)]["sigmin_SG"]
            sig.max                     = self.priors[str(constraint_option)]["sigmax_SG"]
           
            
            model_dict = {
                y: self.priors[str(constraint_option)]["cont_lvl"] - A*(exp(-(w**2)/2.)).as_expr()
            }

            model = Model(model_dict)
            if(verbose):
                print(model)

            # Perform the fit - sigma_y is needed for chi^2 optimization
            fit = Fit(
                model,
                x         = xdata,
                y         = ydata,
                sigma_y   = final_error_copy,
                minimizer = BasinHopping
            )


        fit_result = fit.execute(seed=int(time.time())+seed,stepsize=stepsize)
        # zfit = model(x = xdata, **fit_result.params)

        if(verbose):
            print(fit_result)

        if(double_gaussian==True):
            A_out         = fit_result.params['A']
            A_out_sig     = fit_result.stdev(A)

            A1_out        = fit_result.params['A1']
            A1_out_sig    = fit_result.stdev(A1)

            sig_out       = fit_result.params['sig']
            sig_out_sig   = fit_result.stdev(sig)

            sig1_out      = fit_result.params['sig1']
            sig1_out_sig  = fit_result.stdev(sig1)

            v_out         = fit_result.params['v']
            v_out_sig     = fit_result.stdev(v)

            v1_out        = fit_result.params['v1']
            v1_out_sig    = fit_result.stdev(v1)

            ### Deconvolved absorption profile values
            if(self.inst_sigma>sig_out or self.inst_sigma>sig1_out):
                raise ValueError("Chosen instrument resolution will be unable to resolve the ISM line. Please check the chosen instrument sigma (km/s)")

            sig_out_deconv  = np.sqrt(sig_out**2 - self.inst_sigma**2)
            sig1_out_deconv = np.sqrt(sig1_out**2 - self.inst_sigma**2)

            A_out_deconv    = A_out*(sig_out/sig_out_deconv)
            A1_out_deconv   = A1_out*(sig1_out/sig1_out_deconv)

            v_out_deconv    = v_out
            v1_out_deconv   = v1_out

            ## double gaussian function
            y00 = [ -A_out*np.exp(-0.5*(v_out -i)**2/sig_out**2) - A1_out*np.exp(-0.5*(v1_out - i)**2/sig1_out**2) + self.priors[str(constraint_option)]["cont_lvl"] for i in wav_copy]
            ## First component
            y01 = [ -A_out*np.exp(-0.5*(v_out -i)**2/sig_out**2) + self.priors[str(constraint_option)]["cont_lvl"] for i in wav_copy ]
            ## Second component
            y02 = [ -A1_out*np.exp(-0.5*(v1_out - i)**2/sig1_out**2) + self.priors[str(constraint_option)]["cont_lvl"] for i in wav_copy]

            chi_sq = np.sum(((np.array(y00)-final_profile_copy)**2)/final_error_copy**2,axis=0)


        elif(double_gaussian==False):
            A_out        = fit_result.params['A']
            A_out_sig    = fit_result.stdev(A)

            A1_out       = 0
            A1_out_sig   = 0

            sig_out      = fit_result.params['sig']
            sig_out_sig  = fit_result.stdev(sig)

            sig1_out     = 0
            sig1_out_sig = 0

            v_out        = fit_result.params['v']
            v_out_sig    = fit_result.stdev(v)

            v1_out       = 0
            v1_out_sig   = 0

            ### Deconvolved absorption profile values
            if(self.inst_sigma>sig_out):
                raise ValueError("Chosen instrument resolution will be unable to resolve the ISM line. Please check the chosen instrument sigma (km/s)")


            sig_out_deconv  = np.sqrt(sig_out**2 - self.inst_sigma**2)
            sig1_out_deconv = 0

            A_out_deconv    = A_out*(sig_out/sig_out_deconv)
            A1_out_deconv   = 0

            v_out_deconv    = v_out
            v1_out_deconv   = 0


            ## double gaussian function
            y00 = [ -A_out*np.exp(-0.5*(v_out -i)**2/sig_out**2) + self.priors[str(constraint_option)]["cont_lvl"] for i in wav_copy]
            ## First component
            y01 = [ -A_out*np.exp(-0.5*(v_out -i)**2/sig_out**2) + self.priors[str(constraint_option)]["cont_lvl"] for i in wav_copy]
            ## Second component
            y02 = [ self.priors[str(constraint_option)]["cont_lvl"] for i in xdata]

            chi_sq = np.sum(((np.array(y00)-final_profile_copy)**2)/final_error_copy**2,axis=0)

        ### Pack all results
        self.results_dict = {
            'niter'              : niter,
            'seed'               : seed,
            'cont_lvl'           : self.priors[str(constraint_option)]["cont_lvl"], ## Continuum level
            'constraint_option'  : constraint_option,                               ## which constraint option was used
            'fit_priors_used'    : self.priors[str(constraint_option)],             ## Copy of all the priors used.
            'A_out'              : A_out,
            'A_out_sig'          : A_out_sig,
            'A1_out'             : A1_out,
            'A1_out_sig'         : A1_out_sig,
            'sig_out'            : sig_out,
            'sig_out_sig'        : sig_out_sig,
            'sig1_out'           : sig1_out,
            'sig1_out_sig'       : sig1_out_sig,
            'v_out'              : v_out,
            'v_out_sig'          : v_out_sig,
            'v1_out'             : v1_out,
            'v1_out_sig'         : v1_out_sig,
            'chi_sq'             : chi_sq - chi_sq_continuum,
            'chi_sq_cont'        : chi_sq_continuum,
            
            'fitted_wav'         : wav_copy,
            'fitted_flux'        : y00,
            'fitted_flux_comp1'  : y01,
            'fitted_flux_comp2'  : y02,

            'A_out_deconv'       : A_out_deconv,
            'A1_out_deconv'      : A1_out_deconv,
            'sig_out_deconv'     : sig_out_deconv,
            'sig1_out_deconv'    : sig1_out_deconv,
            'v_out_deconv'       : v_out_deconv,
            'v1_out_deconv'      : v1_out_deconv,


        }

        self.results_dict['residual']          = np.sum((np.array(y00) - final_profile_copy)**2,axis=0)

        self.results_dict['initial-values']    = self.init_values

        self.results_dict['derived_results']   = self.derivedVelocityMeasurements(
            A_out,
            A1_out,
            v_out,
            v1_out,
            sig_out,
            sig1_out,
            double_gaussian
        )

        self.results_dict['derived_results_deconv'] = self.derivedVelocityMeasurements(
            A_out_deconv,
            A1_out_deconv,
            v_out_deconv,
            v1_out_deconv,
            sig_out_deconv,
            sig1_out_deconv,
            double_gaussian
        )


        self.results_dict['derived_results_onlyoutflow'] = self.derivedVelocityMeasurements(
            A_out,
            A1_out,
            v_out,
            v1_out,
            sig_out,
            sig1_out,
            double_gaussian,
            outflow=True
        )

        return(self.results_dict)