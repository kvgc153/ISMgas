import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from symfit import variables, parameters, Model, Fit, exp, GreaterThan, LessThan
from symfit.core.minimizers import BasinHopping

import time
from ISMgas.fitting.init_fitting import *
from ISMgas.SupportingFunctions import load_pickle, save_as_pickle, plotHistogram, plotWithError, getNestedArrayValue
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
                 
                print("Chosen instrument resolution will be unable to resolve the ISM line. Please check the chosen instrument sigma (km/s)")
                return({})

            else:
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
                print("Chosen instrument resolution will be unable to resolve the ISM line. Please check the chosen instrument sigma (km/s)")
                return({})
                

            else:
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
    
    @staticmethod
    def fitGaussian(
        specDict, 
        seedsInit         = 12345,
        nseeds            = 5,
        qtys = [
            ['derived_results','v50'], 
            ['derived_results','v25'], 
            ['derived_results','v05'], 
            ['derived_results','delta_v90'], 
            ['derived_results','EW_kms']
        ], 
        double_gaussian   = True,
        constraint_option = 1,
        priors            = [],
        plotting          = True,
        inst_sigma        = 1
    ):    
        '''
        A convenience function to quantify the uncertainity in the DG/SG fits.
        This fits the absorption profile using multiple seeds and computes
        the median and MAD values for all the quantities (e.g., v50, deltav90).
        '''

        xChosen    = specDict['wav']
        yChosen    = specDict['flux']
        yErrChosen = specDict['error']

        fitResults = []

        if(plotting):
            plt.figure()
            plotWithError(
                x = xChosen,
                y = yChosen, 
                yerr = yErrChosen
            )

        for seeds in np.arange(seedsInit,seedsInit+nseeds,1):
            
            dd = DoubleGaussian(
                x    = xChosen,
                y    = yChosen, 
                yerr = yErrChosen,
                inst_sigma = inst_sigma
            )
            
            if(len(priors)>0):
                ## manually set the priors if needed
                dd.priors = priors
            
            results  = dd.fitting(
                constraint_option = constraint_option, 
                double_gaussian   = double_gaussian, 
                seed = seeds
            )
            
            if(len(results)>0): 
                ## If fitting fails due to instrument resolution issues i.e., results = {} then ignore and continue.
                
                fitResults.append(results)

                if(plotting):
                    plt.plot(
                        results['fitted_wav'],
                        results['fitted_flux'],
                        color = 'purple',
                        alpha = 0.5
                    )

        resultsDict = {}
        for qty in qtys:
            resultsDict[qty[-1]] = plotHistogram(
                [getNestedArrayValue(i, qty) for i in fitResults], 
                plotting=False
            )
            
        return(resultsDict)


##################
### kvgc code ####
##################

class unWrapFittingResults:
    '''
    This class is used to combine the different results from the DoubleGaussian class
    '''
    def __init__(self,seeds,double_gaussian,folderName,objid):
        self.seeds              = seeds
        self.double_gaussian    = double_gaussian
        self.folderName         = folderName
        self.obj_id             = objid
        
        self.dataloader()
        self.computeResults()

    def dataloader(self):
        if(self.double_gaussian==True):
            self.suffix = "DG"
            foo11 = []
            for foo10 in self.seeds:
                foo11.append(load_pickle(self.folderName+"/"+self.obj_id+"_"+str(foo10)+".pkl"))


        elif(self.double_gaussian==False):
            foo11 = []
            self.suffix = "SG"

            for foo10 in self.seeds:
                foo11.append(load_pickle(self.folderName +"/SG-"+self.obj_id+"_"+str(foo10)+".pkl"))

        self.chi_sq        = np.array([float(i['chi_sq']) for i in foo11])
        self.cont_lvl      = float(foo11[0]['cont_lvl'])

#         chi_threshold = 0

        self.init_A         = np.array([float(i['initial-values']['A']) for i in foo11])
        self.init_A1        = np.array([float(i['initial-values']['A1']) for i in foo11])
        self.init_sig       = np.array([float(i['initial-values']['sig']) for i in foo11])
        self.init_sig1      = np.array([float(i['initial-values']['sig1']) for i in foo11])
        self.init_v         = np.array([float(i['initial-values']['v']) for i in foo11])
        self.init_v1        = np.array([float(i['initial-values']['v1']) for i in foo11])

        self.A_out         = np.array([float(i['A_out']) for i in foo11])
        self.A1_out        = np.array([float(i['A1_out']) for i in foo11])
        self.sig_out       = np.array([float(i['sig_out']) for i in foo11])
        self.sig1_out      = np.array([float(i['sig1_out']) for i in foo11])
        self.v_out         = np.array([float(i['v_out']) for i in foo11])
        self.v1_out        = np.array([float(i['v1_out']) for i in foo11])


        self.A_out_deconv         = np.array([float(i['A_out_deconv']) for i in foo11])
        self.A1_out_deconv        = np.array([float(i['A1_out_deconv']) for i in foo11])
        self.sig_out_deconv       = np.array([float(i['sig_out_deconv']) for i in foo11])
        self.sig1_out_deconv      = np.array([float(i['sig1_out_deconv']) for i in foo11])
        self.v_out_deconv         = np.array([float(i['v_out_deconv']) for i in foo11])
        self.v1_out_deconv        = np.array([float(i['v1_out_deconv']) for i in foo11])


        self.residual      = np.array([float(i['residual']) for i in foo11])

        self.residual_out, self.residual_outsig =  plotHistogram(self.residual,arrayName       = "Residual", plotting = False)


        self.v01_out       = np.array([float(i['derived_results']['v01']) for i in foo11])
        self.v05_out       = np.array([float(i['derived_results']['v05']) for i in foo11])
        self.v10_out       = np.array([float(i['derived_results']['v10']) for i in foo11])
        self.v25_out       = np.array([float(i['derived_results']['v25']) for i in foo11])
        self.v50_out       = np.array([float(i['derived_results']['v50']) for i in foo11])
        self.v75_out       = np.array([float(i['derived_results']['v75']) for i in foo11])
        self.v90_out       = np.array([float(i['derived_results']['v90']) for i in foo11])
        self.v95_out       = np.array([float(i['derived_results']['v95']) for i in foo11])
        self.v99_out       = np.array([float(i['derived_results']['v99']) for i in foo11])
        self.delta_v98_out = np.array([float(i['derived_results']['delta_v98']) for i in foo11])
        self.delta_v90_out = np.array([float(i['derived_results']['delta_v90']) for i in foo11])
        self.delta_v80_out = np.array([float(i['derived_results']['delta_v80']) for i in foo11])
        self.delta_v50_out = np.array([float(i['derived_results']['delta_v50']) for i in foo11])

        self.v01_out_deconv       = np.array([float(i['derived_results_deconv']['v01']) for i in foo11])
        self.v05_out_deconv       = np.array([float(i['derived_results_deconv']['v05']) for i in foo11])
        self.v10_out_deconv       = np.array([float(i['derived_results_deconv']['v10']) for i in foo11])
        self.v25_out_deconv       = np.array([float(i['derived_results_deconv']['v25']) for i in foo11])
        self.v50_out_deconv       = np.array([float(i['derived_results_deconv']['v50']) for i in foo11])
        self.v75_out_deconv       = np.array([float(i['derived_results_deconv']['v75']) for i in foo11])
        self.v90_out_deconv       = np.array([float(i['derived_results_deconv']['v90']) for i in foo11])
        self.v95_out_deconv       = np.array([float(i['derived_results_deconv']['v95']) for i in foo11])
        self.v99_out_deconv       = np.array([float(i['derived_results_deconv']['v99']) for i in foo11])
        self.delta_v98_out_deconv = np.array([float(i['derived_results_deconv']['delta_v98']) for i in foo11])
        self.delta_v90_out_deconv = np.array([float(i['derived_results_deconv']['delta_v90']) for i in foo11])
        self.delta_v80_out_deconv = np.array([float(i['derived_results_deconv']['delta_v80']) for i in foo11])
        self.delta_v50_out_deconv = np.array([float(i['derived_results_deconv']['delta_v50']) for i in foo11])


        self.v01_out1       = np.array([float(i['derived_results_onlyoutflow']['v01']) for i in foo11])
        self.v05_out1       = np.array([float(i['derived_results_onlyoutflow']['v05']) for i in foo11] )
        self.v10_out1       = np.array([float(i['derived_results_onlyoutflow']['v10']) for i in foo11])
        self.v25_out1       = np.array([float(i['derived_results_onlyoutflow']['v25']) for i in foo11])
        self.v50_out1       = np.array([float(i['derived_results_onlyoutflow']['v50']) for i in foo11])
        self.v75_out1       = np.array([float(i['derived_results_onlyoutflow']['v75']) for i in foo11])
        self.v90_out1       = np.array([float(i['derived_results_onlyoutflow']['v90']) for i in foo11])
        self.v95_out1       = np.array([float(i['derived_results_onlyoutflow']['v95']) for i in foo11])
        self.v99_out1       = np.array([float(i['derived_results_onlyoutflow']['v99']) for i in foo11])
        self.delta_v98_out1 = np.array([float(i['derived_results_onlyoutflow']['delta_v98']) for i in foo11])
        self.delta_v90_out1 = np.array([float(i['derived_results_onlyoutflow']['delta_v90']) for i in foo11])
        self.delta_v80_out1 = np.array([float(i['derived_results_onlyoutflow']['delta_v80']) for i in foo11])
        self.delta_v50_out1 = np.array([float(i['derived_results_onlyoutflow']['delta_v50']) for i in foo11] )



    def computeResults(self):

        ###########################################
        plt.figure(figsize=(13,9))
        suptitle =plt.suptitle(f"Initial parameters-${self.suffix}",y=1.05,fontsize=20)
        xnums = 3
        ynums = 2

        plt.subplot(ynums,xnums,1)
        plotHistogram(self.init_A, best_fit_plot = False, arrayName = "$A$")

        plt.subplot(ynums,xnums,2)
        plotHistogram(self.init_A1, best_fit_plot = False, arrayName = "$A_1$",)

        plt.subplot(ynums,xnums,3)
        plotHistogram(self.init_sig, best_fit_plot = False, arrayName = "$\sigma$")

        plt.subplot(ynums,xnums,4)
        plotHistogram(self.init_sig1, best_fit_plot = False, arrayName = "$\sigma_1$")

        plt.subplot(ynums,xnums,5)
        plotHistogram(self.init_v, best_fit_plot   = False, arrayName = "$v$")

        plt.subplot(ynums,xnums,6)
        plotHistogram(self.init_v1, best_fit_plot   = False, arrayName = "$v_1$")
        
        plt.tight_layout()
        plt.savefig(f"Initial-fits-histogram-{self.suffix}.png", bbox_extra_artists=(suptitle,), **ISMgasPlot['savefig'])
        plt.close()
        
        ##########################################



        ###########################################
        plt.figure(figsize=(13,9))
        suptitle =plt.suptitle(f"Final Fit parameters-{self.suffix}",y=1.05,fontsize=20)

        xnums = 3
        ynums = 2

        plt.subplot(ynums,xnums,1)
        self.A, self.Asig           = plotHistogram(self.A_out, arrayName = "$A$")

        plt.subplot(ynums,xnums,2)
        self.A1, self.A1sig         = plotHistogram(self.A1_out, arrayName = "$A_1$")

        plt.subplot(ynums,xnums,3)
        self.sigma, self.sigmasig   = plotHistogram(self.sig_out, arrayName = "$\sigma$")

        plt.subplot(ynums,xnums,4)
        self.sigma1, self.sigma1sig = plotHistogram(self.sig1_out, arrayName = "$\sigma_1$")

        plt.subplot(ynums,xnums,5)
        self.v, self.vsig           = plotHistogram(self.v_out, arrayName = "$v$")

        plt.subplot(ynums,xnums,6)
        self.v1, self.v1sig         = plotHistogram(self.v1_out, arrayName = "$v_1$")

        plt.tight_layout()
        plt.savefig(f"All-fits-histogram-{self.suffix}.png", bbox_extra_artists=(suptitle,), **ISMgasPlot['savefig'])
        plt.close()
        ###########################################


        ###########################################
        plt.figure(figsize=(13,9))
        suptitle =plt.suptitle(f"Final Fit parameters deconvolved-{self.suffix}",y=1.05,fontsize=20)
        xnums = 3
        ynums = 2


        plt.subplot(ynums,xnums,1)
        self.A_deconv,self.Asig_deconv            = plotHistogram(self.A_out_deconv, arrayName = "$A$")

        plt.subplot(ynums,xnums,2)
        self.A1_deconv,self.A1sig_deconv          = plotHistogram(self.A1_out_deconv, arrayName = "$A_1$")

        plt.subplot(ynums,xnums,3)
        self.sigma_deconv,self.sigmasig_deconv    = plotHistogram(self.sig_out_deconv, arrayName = "$\sigma$")

        plt.subplot(ynums,xnums,4)
        self.sigma1_deconv,self.sigma1sig_deconv  = plotHistogram(self.sig1_out_deconv, arrayName = "$\sigma_1$")

        plt.subplot(ynums,xnums,5)
        self.v_deconv,self.vsig_deconv            = plotHistogram(self.v_out_deconv, arrayName = "$v$")

        plt.subplot(ynums,xnums,6)
        self.v1_deconv,self.v1sig_deconv          = plotHistogram(self.v1_out_deconv, arrayName = "$v_1$")

        plt.tight_layout()
        plt.savefig(f"All-fits-deconvolved-histogram-{self.suffix}.png",dpi=50,bbox_extra_artists=(suptitle,), bbox_inches="tight")
        plt.close()
        ###########################################


        ###########################################
        plt.figure(figsize=(18,18))
        suptitle =plt.suptitle(f"Derived velocity parameters-{self.suffix}",y=1.05,fontsize=25)
        
        plt.subplot(4,4,1)
        self.v01,self.v01sig    = plotHistogram(self.v01_out, arrayName = "$v_{01}$")

        plt.subplot(4,4,2)
        self.v05,self.v05sig    = plotHistogram(self.v05_out, arrayName    = "$v_{05}$")

        plt.subplot(4,4,3)
        self.v10,self.v10sig    = plotHistogram(self.v10_out, arrayName    = "$v_{10}$")

        plt.subplot(4,4,4)
        self.v25,self.v25sig    = plotHistogram(self.v25_out, arrayName    = "$v_{25}$")

        plt.subplot(4,4,5)
        self.v99,self.v99sig    = plotHistogram(self.v99_out, arrayName    = "$v_{99}$")

        plt.subplot(4,4,6)
        self.v95,self.v95sig    = plotHistogram(self.v95_out, arrayName    = "$v_{95}$")

        plt.subplot(4,4,7)
        self.v90,self.v90sig    = plotHistogram(self.v90_out, arrayName    = "$v_{90}$")

        plt.subplot(4,4,8)
        self.v75,self.v75sig    = plotHistogram(self.v75_out, arrayName    = "$v_{75}$")

        plt.subplot(4,4,9)
        self.delta_v98,self.delta_v98sig    = plotHistogram(self.delta_v98_out, arrayName    = "$\Delta v_{98}$")

        plt.subplot(4,4,10)
        self.delta_v90,self.delta_v90sig    = plotHistogram(self.delta_v90_out, arrayName = "$\Delta v_{90}$")

        plt.subplot(4,4,11)
        self.delta_v80,self.delta_v80sig    = plotHistogram(self.delta_v80_out, arrayName = "$\Delta v_{80}$")

        plt.subplot(4,4,12)
        self.delta_v50,self.delta_v50sig    = plotHistogram(self.delta_v50_out, arrayName = "$\Delta v_{50}$")

        plt.subplot(4,4,13)
        self.v50,self.v50sig                = plotHistogram(self.v50_out, arrayName    = "$v_{50}$")

        plt.tight_layout()

        plt.savefig(f"derived-measurements-histogram-{self.suffix}.png",dpi=50,bbox_extra_artists=(suptitle,), bbox_inches="tight")
        plt.close()

        ###########################################


        ###########################################
        plt.figure(figsize=(18,18))
        suptitle =plt.suptitle(f"Derived velocity deconvolved parameters-{self.suffix}",y=1.05,fontsize=25)
        
        plt.subplot(4,4,1)
        self.v01_deconv,self.v01sig_deconv = plotHistogram(self.v01_out_deconv, arrayName    = "$v_{01}$")

        plt.subplot(4,4,2)
        self.v05_deconv,self.v05sig_deconv  =plotHistogram(self.v05_out_deconv, arrayName    = "$v_{05}$")

        plt.subplot(4,4,3)
        self.v10_deconv,self.v10sig_deconv  =plotHistogram(self.v10_out_deconv, arrayName    = "$v_{10}$")


        plt.subplot(4,4,4)
        self.v25_deconv,self.v25sig_deconv  =plotHistogram(self.v25_out_deconv, arrayName    = "$v_{25}$")

        plt.subplot(4,4,5)
        self.v99_deconv,self.v99sig_deconv  =plotHistogram(self.v99_out_deconv, arrayName    = "$v_{99}$")

        plt.subplot(4,4,6)
        self.v95_deconv,self.v95sig_deconv  =plotHistogram(self.v95_out_deconv, arrayName    = "$v_{95}$")

        plt.subplot(4,4,7)
        self.v90_deconv,self.v90sig_deconv  =plotHistogram(self.v90_out_deconv, arrayName    = "$v_{90}$")
        
        plt.subplot(4,4,8)
        self.v75_deconv,self.v75sig_deconv  =plotHistogram(self.v75_out_deconv, arrayName    = "$v_{75}$")

        plt.subplot(4,4,9)
        self.delta_v98_deconv,self.delta_v98sig_deconv  =plotHistogram(self.delta_v98_out_deconv, arrayName    = "$\Delta v_{98}$")

        plt.subplot(4,4,10)
        self.delta_v90_deconv,self.delta_v90sig_deconv   =plotHistogram(self.delta_v90_out_deconv, arrayName = "$\Delta v_{90}$")

        plt.subplot(4,4,11)
        self.delta_v80_deconv,self.delta_v80sig_deconv   =plotHistogram(self.delta_v80_out_deconv, arrayName = "$\Delta v_{80}$")

        plt.subplot(4,4,12)
        self.delta_v50_deconv,self.delta_v50sig_deconv   =plotHistogram(self.delta_v50_out_deconv, arrayName = "$\Delta v_{50}$")

        plt.subplot(4,4,13)
        self.v50_deconv,self.v50sig_deconv  =plotHistogram(self.v50_out_deconv, arrayName    = "$v_{50}$")

        plt.tight_layout()
        plt.savefig(f"derived-measurements-deconvolved-histogram-{self.suffix}.png", bbox_extra_artists=(suptitle,), **ISMgasPlot['savefig'])
        plt.close()

        ##########################################

        ##########################################
        plt.figure(figsize=(18,18))
        suptitle =plt.suptitle(f"Derived velocity parameters-{self.suffix}: Only outflows",y=1.05,fontsize=25)
        
        plt.subplot(4,4,1)
        self.v01_1,self.v01sig_1    = plotHistogram(self.v01_out1, arrayName = "$v_{01}$")

        plt.subplot(4,4,2)
        self.v05_1,self.v05sig_1    = plotHistogram(self.v05_out1, arrayName    = "$v_{05}$")

        plt.subplot(4,4,3)
        self.v10_1,self.v10sig_1    = plotHistogram(self.v10_out1, arrayName    = "$v_{10}$")

        plt.subplot(4,4,4)
        self.v25_1,self.v25sig_1    = plotHistogram(self.v25_out1, arrayName    = "$v_{25}$")

        plt.subplot(4,4,5)
        self.v99_1,self.v99sig_1    = plotHistogram(self.v99_out1, arrayName    = "$v_{99}$")

        plt.subplot(4,4,6)
        self.v95_1,self.v95sig_1    = plotHistogram(self.v95_out1, arrayName    = "$v_{95}$")

        plt.subplot(4,4,7)
        self.v90_1,self.v90sig_1    = plotHistogram(self.v90_out1, arrayName    = "$v_{90}$")

        plt.subplot(4,4,8)
        self.v75_1,self.v75sig_1    = plotHistogram(self.v75_out1, arrayName    = "$v_{75}$")

        plt.subplot(4,4,9)
        self.delta_v98_1,self.delta_v98sig_1    = plotHistogram(self.delta_v98_out1, arrayName    = "$\Delta v_{98}$")

        plt.subplot(4,4,10)
        self.delta_v90_1,self.delta_v90sig_1    = plotHistogram(self.delta_v90_out1, arrayName = "$\Delta v_{90}$")

        plt.subplot(4,4,11)
        self.delta_v80_1,self.delta_v80sig_1    = plotHistogram(self.delta_v80_out1, arrayName = "$\Delta v_{80}$")

        plt.subplot(4,4,12)
        self.delta_v50_1,self.delta_v50sig_1    = plotHistogram(self.delta_v50_out1, arrayName = "$\Delta v_{50}$")

        plt.subplot(4,4,13)
        self.v50_1,self.v50sig_1                = plotHistogram(self.v50_out1, arrayName    = "$v_{50}$")
        
        plt.tight_layout()
        plt.savefig(f"derived-measurements-onlyoutflow-histogram-{self.suffix}.png", bbox_extra_artists=(suptitle,), **ISMgasPlot['savefig'])
        plt.close()
        
        ###########################################


    def saveResults(self):
        resultDict = {
            'objid'         : self.obj_id,
            'A_out'         : self.A,
            'A_out_sig'     : self.Asig,
            'A1_out'        : self.A1,
            'A1_out_sig'    : self.A1sig,
            'sig_out'       : self.sigma,
            'sig_out_sig'   : self.sigmasig,
            'sig1_out'      : self.sigma1,
            'sig1_out_sig'  : self.sigma1sig,
            'v_out'         : self.v,
            'v_out_sig'     : self.vsig,
            'v1_out'        : self.v1,
            'v1_out_sig'    : self.v1sig,
            'v01'           : self.v01,
            'v01sig'        : self.v01sig,
            'v05'           : self.v05,
            'v05sig'        : self.v05sig,
            'v10'           : self.v10,
            'v10sig'        : self.v10sig,
            'v25'           : self.v25,
            'v25sig'        : self.v25sig,
            'v99'           : self.v99,
            'v99sig'        : self.v99sig,
            'v95'           : self.v95,
            'v95sig'        : self.v95sig,
            'v90'           : self.v90,
            'v90sig'        : self.v90sig,
            'v75'           : self.v75,
            'v75sig'        : self.v75sig,
            'delta_v98'     : self.delta_v98,
            'delta_v98sig'  : self.delta_v98sig,
            'delta_v90'     : self.delta_v90,
            'delta_v90sig'  : self.delta_v90sig,
            'delta_v80'     : self.delta_v80,
            'delta_v80sig'  : self.delta_v80sig,
            'delta_v50'     : self.delta_v50,
            'delta_v50sig'  : self.delta_v50sig,
            'v50'           : self.v50,
            'v50sig'        : self.v50sig,

            'A_out_deconv'         : self.A_deconv,
            'A_out_sig_deconv'     : self.Asig_deconv,
            'A1_out_deconv'        : self.A1_deconv,
            'A1_out_sig_deconv'    : self.A1sig_deconv,
            'sig_out_deconv'       : self.sigma_deconv,
            'sig_out_sig_deconv'   : self.sigmasig_deconv,
            'sig1_out_deconv'      : self.sigma1_deconv,
            'sig1_out_sig_deconv'  : self.sigma1sig_deconv,
            'v_out_deconv'         : self.v_deconv,
            'v_out_sig_deconv'     : self.vsig_deconv,
            'v1_out_deconv'        : self.v1_deconv,
            'v1_out_sig_deconv'    : self.v1sig_deconv,
            'v01_deconv'           : self.v01_deconv,
            'v01sig_deconv'        : self.v01sig_deconv,
            'v05_deconv'           : self.v05_deconv,
            'v05sig_deconv'        : self.v05sig_deconv,
            'v10_deconv'           : self.v10_deconv,
            'v10sig_deconv'        : self.v10sig_deconv,
            'v25_deconv'           : self.v25_deconv,
            'v25sig_deconv'        : self.v25sig_deconv,
            'v99_deconv'           : self.v99_deconv,
            'v99sig_deconv'        : self.v99sig_deconv,
            'v95_deconv'           : self.v95_deconv,
            'v95sig_deconv'        : self.v95sig_deconv,
            'v90_deconv'           : self.v90_deconv,
            'v90sig_deconv'        : self.v90sig_deconv,
            'v75_deconv'           : self.v75_deconv,
            'v75sig_deconv'        : self.v75sig_deconv,
            'delta_v98_deconv'     : self.delta_v98_deconv,
            'delta_v98sig_deconv'  : self.delta_v98sig_deconv,
            'delta_v90_deconv'     : self.delta_v90_deconv,
            'delta_v90sig_deconv'  : self.delta_v90sig_deconv,
            'delta_v80_deconv'     : self.delta_v80_deconv,
            'delta_v80sig_deconv'  : self.delta_v80sig_deconv,
            'delta_v50_deconv'     : self.delta_v50_deconv,
            'delta_v50sig_deconv'  : self.delta_v50sig_deconv,
            'v50_deconv'           : self.v50_deconv,
            'v50sig_deconv'        : self.v50sig_deconv,


            'v01_1'         : self.v01_1,
            'v01sig_1'      : self.v01sig_1,
            'v05_1'         : self.v05_1,
            'v05sig_1'      : self.v05sig_1,
            'v10_1'         : self.v10_1,
            'v10sig_1'      : self.v10sig_1,
            'v25_1'         : self.v25_1,
            'v25sig_1'      : self.v25sig_1,
            'v99'           : self.v99,
            'v99sig_1'      : self.v99sig_1,
            'v95'           : self.v95,
            'v95sig_1'      : self.v95sig_1,
            'v90'           : self.v90,
            'v90sig_1'      : self.v90sig_1,
            'v75'           : self.v75,
            'v75sig_1'      : self.v75sig_1,
            'delta_v98_1'   : self.delta_v98_1,
            'delta_v98sig_1': self.delta_v98sig_1,
            'delta_v90_1'   : self.delta_v90_1,
            'delta_v90sig_1': self.delta_v90sig_1,
            'delta_v80_1'   : self.delta_v80_1,
            'delta_v80sig_1': self.delta_v80sig_1,
            'delta_v50_1'   : self.delta_v50_1,
            'delta_v50sig_1': self.delta_v50sig_1,
            'v50_1'         : self.v50_1,
            'v50sig_1'      : self.v50sig_1,
            'residual'      : self.residual_out,
            'residual_sig'  : self.residual_outsig,
            'cont_lvl'      : self.cont_lvl,
            'initial-values' : [],
            'derived_results': [],
            'derived_results_deconv': [],
            'derived_results_onlyoutflow': [],


        }
        ## save results
        save_as_pickle(
                arrayToSave   = resultDict,
                fileName      = self.obj_id+f"-{self.suffix}-Results.pkl"
        )

        return(resultDict)




class plotFittingResults:

    def __init__(self,**kwargs):
        self.DG_dict    = kwargs.get("DG_dict")
        self.SG_dict    = kwargs.get("SG_dict")
        self.spec_dict  = kwargs.get("spec_dict")

        self.objid       =  self.DG_dict["objid"]

        self.cont_lvl    =  self.DG_dict["cont_lvl"]
        self.A           =  self.DG_dict['A_out']
        self.A1          =  self.DG_dict['A1_out']
        self.sigma       =  self.DG_dict['sig_out']
        self.sigma1      =  self.DG_dict['sig1_out']
        self.v           =  self.DG_dict['v_out']
        self.v1          =  self.DG_dict['v1_out']

        self.A_deconv           =  self.DG_dict['A_out_deconv']
        self.A1_deconv          =  self.DG_dict['A1_out_deconv']
        self.sigma_deconv       =  self.DG_dict['sig_out_deconv']
        self.sigma1_deconv      =  self.DG_dict['sig1_out_deconv']
        self.v_deconv           =  self.DG_dict['v_out_deconv']
        self.v1_deconv          =  self.DG_dict['v1_out_deconv']


        self.v25_1       =  self.DG_dict['v25_1']


        self.A_SG        =  self.SG_dict['A_out']
        self.A1_SG       =  self.SG_dict['A1_out']
        self.sigma_SG    =  self.SG_dict['sig_out']
        self.sigma1_SG   =  self.SG_dict['sig1_out']
        self.v_SG        =  self.SG_dict['v_out']
        self.v1_SG       =  self.SG_dict['v1_out']

        self.source_wav     = self.spec_dict['source_wav']
        self.source_flux    = self.spec_dict['source_flux']
        self.source_error   = self.spec_dict['source_error']
        self.spectra_interp = self.spec_dict['spectra_interp']
        self.sigma_lines    = self.spec_dict['sigma']
        self.line_name      = self.spec_dict['line_name']


        ## double gaussian function
        self.DGFit          = [ -self.A*np.exp(-0.5*(self.v -i)**2/self.sigma**2) - self.A1*np.exp(-0.5*(self.v1 - i)**2/self.sigma1**2) + self.cont_lvl for i in self.source_wav]

        self.DGFit_deconv   = [ -self.A_deconv*np.exp(-0.5*(self.v_deconv -i)**2/self.sigma_deconv**2) - self.A1_deconv*np.exp(-0.5*(self.v1_deconv - i)**2/self.sigma1_deconv**2) + self.cont_lvl for i in self.source_wav]

        ## First component
        self.DGFitComp1     = [ -self.A*np.exp(-0.5*(self.v -i)**2/self.sigma**2) + self.cont_lvl  for i in self.source_wav]
        ## Second component
        self.DGFitComp2     = [ - self.A1*np.exp(-0.5*(self.v1 - i)**2/self.sigma1**2) + self.cont_lvl  for i in self.source_wav ]
        ## single gaussian fit component
        self.SGFit          = [ - self.A_SG*np.exp(-0.5*(self.v_SG - i)**2/self.sigma_SG**2) + self.cont_lvl  for i in self.source_wav ]



    def plotVelocityFlux(self,**kwargs):
        '''
        kwargs:
        - vline:
        - xlim:
        - ylim:
        - xticks:
        - yticks:
        - fontsize:
        - vline: Default is it plots v25_1
        '''
        vline     = kwargs.get('vline',[self.v25_1])
        font_size = kwargs.get('fontsize', 50)

        fig1 = plt.figure(figsize=(30,19))

        ## Wavelength v/s flux plot
        plt.plot(
            self.source_wav,
            self.source_flux,
            alpha = 0.8,
            linewidth=15,
            label='data',
            color = color_pal_kvgc['pub1'][28]
        )



        plt.fill_between(
            self.source_wav,
            self.source_flux-2*self.source_error,
            self.source_flux+2*self.source_error,
            alpha=0.8,
            facecolor=color_pal_kvgc['pub1'][4]
        )

        ## Fitted flux plot
        plt.plot(
            self.source_wav,
            self.DGFit,
            linewidth=10,
            alpha = 1,
            color = color_pal_kvgc['pub1'][13],
            label='fit'
        )



        plt.plot(
            self.source_wav,
            self.SGFit,
            linewidth=6,
            linestyle='--',
            alpha = 0.7,
            color = color_pal_kvgc['pub1'][8],
            label='fit-SG'
        )

        if(len(vline)>0):
            for foo_vline in vline:
                plt.axvline(
                    foo_vline,
                    linewidth=10,
                    linestyle='--',
                    color='lime',
                    alpha=0.9
                )

        plt.xlim(kwargs.get('xlim',[-1300,1300]))
        plt.ylim(kwargs.get('ylim',[0,None]))

        plt.xticks(kwargs.get('xticks',np.arange(-1000,1001,500)), fontsize=font_size)
        plt.yticks(kwargs.get('yticks',np.arange(0.0,1.4,0.2).round(1)),fontsize=font_size)

        plt.xlabel('Velocity(km/s)',fontsize=font_size)
        plt.ylabel('Flux (Normalized)',fontsize=font_size)

        plt.grid('on',linewidth=5,alpha = 0.6,linestyle='--')

        plt.legend(fontsize=font_size,loc='lower right')
        plt.savefig("%s-outflow-profile.png"%(self.objid), **ISMgasPlot['savefig'])
        plt.close()



        fig1 = plt.figure(figsize=(30,19))
        ## Fitted flux plot

        plt.plot(
            self.source_wav,
            self.source_flux,
            alpha = 0.8,
            linewidth=15,
            label='data',
            color = color_pal_kvgc['pub1'][28]
            )



        plt.fill_between(
            self.source_wav,
            self.source_flux-2*self.source_error,
            self.source_flux+2*self.source_error,
            alpha=0.8,
            facecolor=color_pal_kvgc['pub1'][4]
            )

        plt.plot(
            self.source_wav,
            self.DGFit_deconv,
            linewidth=10,
            alpha = 1,
            color = color_pal_kvgc['pub1'][13],
            label='fit-deconvoled'
            )

        plt.xlim(kwargs.get('xlim',[-1300,1300]))
        plt.ylim(kwargs.get('ylim',[0,None]))

        plt.xticks(kwargs.get('xticks',np.arange(-1000,1001,500)), fontsize=font_size)
        plt.yticks(kwargs.get('yticks',np.arange(0.0,1.4,0.2).round(1)),fontsize=font_size)

        plt.xlabel('Velocity(km/s)',fontsize=font_size)
        plt.ylabel('Flux (Normalized)',fontsize=font_size)

        plt.grid('on',linewidth=5,alpha = 0.6,linestyle='--')

        plt.legend(fontsize=font_size,loc='lower right')
        plt.savefig("%s-outflow-profile-deconvoled.png"%(self.objid), **ISMgasPlot['savefig'])
        plt.close()


    def plotChosenLines(self,**kwargs):
        '''

        '''
        fig = plt.figure(figsize=(30,45))
        font_size = kwargs.get('fontsize',80)

        ##################
        count=len(color_pal_kvgc['pub1'])-1

        for i in reversed(range(len(self.spectra_interp))): ## reversed ensures that the lines match the legend order

            ## Wavelength v/s individual chosen lines
            plt.plot(
                self.source_wav,
                self.spectra_interp[i]+2*i,
                alpha = 0.8,
                linewidth=15,
                label=self.line_name[i],
                color=color_pal_kvgc['pub1'][count]
                )

            e = self.sigma_lines[i] ## Standard error
            ## Removing the high sigma values -- i.e those which were not used for the analysis
            e[e>100] = 0
            plt.fill_between(
                self.source_wav,
                self.spectra_interp[i]+2*i-2*e,
                self.spectra_interp[i]+2*i+2*e,
                alpha=0.7,
                facecolor=color_pal_kvgc['pub1'][4]
                )

            count = count -1


        plt.legend(bbox_to_anchor=(-0.6, 1),
                    loc='upper left',
                    ncol=1,
                    fontsize=font_size)

        plt.xticks(kwargs.get('xticks',np.arange(-2000,2001,500)),
                    fontsize=font_size,
                    rotation=90)

        plt.yticks(kwargs.get('yticks', np.arange(0.0,21,2).round(1)),
                    fontsize=font_size)



        plt.xlabel('Velocity(km/s)',fontsize=font_size)
        plt.ylabel('Flux (Normalized)',fontsize=font_size)

        plt.xlim(kwargs.get('xlim',[-2000,2000]))
        plt.ylim(kwargs.get('ylim',[-0.4,14]))

        plt.grid('on',linewidth=5,alpha = 0.6,linestyle='--')
        plt.savefig("%s-outflow-lines.png"%(self.objid), **ISMgasPlot['savefig'])
        plt.close()





    def plotResiduals(self,**kwargs):
        '''
        kwargs:
        - xlim1 :
        - ylim1 :
        - xticks1 :
        - yticks1 :
        - xtickshist :
        '''
        font_size   = kwargs.get('fontsize',50)

        fig         = plt.figure(figsize=(30,10))
        gs          = gridspec.GridSpec(1, 2, width_ratios=[3, 1])

        plt.subplot(gs[0])

        ## Residual - DG plot
        plt.plot(
            self.source_wav,
            np.array(self.source_flux)-np.array(self.DGFit),
            alpha = 0.8,
            linewidth=10,
            label='Residual',
            color = color_pal_kvgc['pub1'][13]
            )

        ## Residual - SG plot
        plt.plot(
            self.source_wav,
            np.array(self.source_flux)-np.array(self.SGFit),
            alpha = 0.8,
            linewidth=10,
            linestyle='--',
            label='Residual-SG',
            color = color_pal_kvgc['pub1'][8]
            )

        plt.xlabel('Velocity(km/s)',fontsize=font_size)
        plt.ylabel('Residual',fontsize=font_size)

        plt.grid('on',linewidth=5,alpha = 0.6,linestyle='--')

        plt.xticks(kwargs.get('xticks1',np.arange(-1000,1001,500)),
                    fontsize=font_size)
        plt.yticks(kwargs.get('yticks1',np.arange(-1,1,0.2).round(1)),
                    fontsize=font_size)

        plt.legend(fontsize=font_size,loc='lower right')

        plt.xlim(kwargs.get('xlim1',[-1300,1300]))
        plt.ylim(kwargs.get('ylim1',[-0.5,0.5]))


        plt.subplot(gs[1])

        ## Residual -DG histogram
        plt.hist(np.array(self.source_flux)-np.array(self.DGFit),
                    color=color_pal_kvgc['pub1'][13],
                    linewidth=10,
                    alpha= 0.8,
                    bins=np.arange(-1,1,0.02),
                    histtype='step',
                    density=True,
                    orientation="horizontal")

        ## Residual - SG histogram
        plt.hist(np.array(self.source_flux)-np.array(self.SGFit),
                    color=color_pal_kvgc['pub1'][8],
                    linewidth=7,
                    alpha= 0.9,
                    linestyle='--',
                    bins=np.arange(-1,1,0.02),
                    histtype='step',
                    density=True,
                    orientation="horizontal")


        plt.xticks(kwargs.get('xtickshist',np.arange(0,10,1)), fontsize=font_size)
        plt.yticks(fontsize=font_size)

        plt.xlabel("Counts (Normalized)",fontsize=font_size)


        plt.ylim(kwargs.get('ylim1',[-0.5,0.5]))

        plt.grid('on',linewidth=1,alpha = 0.6,linestyle='--')
        plt.tight_layout()

        plt.savefig("%s-residual.png"%(self.objid), **ISMgasPlot['savefig'])

        plt.close()

