import matplotlib.pyplot as plt
import numpy as np  
from ISMgas.linelist import linelist_highz
from ISMgas.SupportingFunctions import plotHistogram
from lmfit import Parameters, Model

def stel1296_complex_model(x, p0, p1, p2, p3, p4, p5, z1, z2, z3, sigma1,sigma2,sigma3, p9, p10):
    """Si-III + ISM absorption line complex model with polynomial continuum. The model includes the following absorption lines:
    - Si III 1294
    - Si III 1296
    - Si III 1298
    - O I 1302
    - Si II 1304

    Args:
        x (_type_): _description_
        p0 (_type_): _description_
        p1 (_type_): _description_
        p2 (_type_): _description_
        p3 (_type_): _description_
        p4 (_type_): _description_
        z1 (_type_): _description_
        z2 (_type_): _description_
        sigma1 (_type_): _description_
        sigma2 (_type_): _description_
        p9 (_type_): _description_
        p10 (_type_): _description_

    Returns:
        _type_: _description_
    """
    zstar = 1. + z1
    zo2 = 1. + z2
    zfineem = 1. + z3

    center = zstar * 0.129889

    y1294 = p0 * np.exp(-0.5 * ((x - zstar * linelist_highz['Si III 1294']['lambda']*1e-4) / sigma1)**2)
    y1296 = p1 * np.exp(-0.5 * ((x - zstar * linelist_highz['Si III 1296']['lambda']*1e-4) / sigma1)**2)
    y1298 = p2 * np.exp(-0.5 * ((x - zstar * linelist_highz['Si III 1298-9']['lambda']*1e-4) / sigma1)**2)

    y1302 = p3 * np.exp(-0.5 * ((x - zo2 * linelist_highz['O I 1302']['lambda']*1e-4) / sigma2)**2) 
    y1304 = p4 * np.exp(-0.5 * ((x - zo2 * linelist_highz['Si II 1304']['lambda']*1e-4) / sigma2)**2)
    
    y1309 = p5 * np.exp(-0.5 * ((x - zfineem * linelist_highz['Si II* 1309']['lambda']*1e-4) / sigma3)**2) ## Fine structure emission line

    poly_cont = -x**3 + p10 * x**2 + p10 * (x - center)  + p9

    return y1294 + y1296 + y1298 + y1302 + y1304 + y1309 + poly_cont


class Fitter:
    def __init__(self, wave, flux, err, zs, objid):
        self.wave = wave ## Wavelength in Angstroms
        self.flux = flux
        self.err = err
        self.zs = zs ## Guess redshift
        self.objid = objid

    def stellar1300(self, waveMin=1285, waveMax=1315, niter=250, plotting=False, priors=None, method='leastsq'):
        
        mask1 = self.wave > waveMin*(1+self.zs)
        mask2 =  self.wave < waveMax*(1+self.zs)
        mask  = mask1 & mask2

        wave_masked = self.wave[mask]
        flux_masked = self.flux[mask]
        err_masked = self.err[mask]
        
        if(plotting):
            plt.figure(figsize=(11,5))

            plt.plot(
                wave_masked, 
                flux_masked,
                lw = 2,
                c = 'black',
                drawstyle='steps-mid'
            )
        zstellar = [] ## Redshift of the stellar component
        zism = [] ## Redshift of the ISM component
        zfineem = [] ## Redshift of the fine structure emission component
        
        sigma_stellar = [] ## Velocity dispersion of the stellar component  
        sigma_ism = [] ## Velocity dispersion of the ISM component
        sigma_fineem = [] ## Velocity dispersion of the fine structure component
        
        ## amplitudes 
        p0 = [] ## Amplitude of Si III 1294
        p1 = [] ## Amplitude of Si III 1296
        p2 = [] ## Amplitude of Si III 1298
        p3 = [] ## Amplitude of O I 1302
        p4 = [] ## Amplitude of Si II 1304
        p5 = [] ## Amplitude of Si II* 1309
        p9 = [] ## Amplitude of the polynomial continuum
        p10 = [] ## Amplitude of the polynomial continuum


        if priors is None:
            priors = {
                'p0':{'value':-0.00001, 'min':-0.04, 'max':0},
                'p1':{'value':-0.00001, 'min':-0.04, 'max':0},
                'p2':{'value':-0.00001, 'min':-0.04, 'max':0},
                'p3':{'value':-0.00001, 'min':-0.05,   'max':0},
                'p4':{'value':-0.001,  'min':-0.05,   'max':0},
                'p5':{'value':0.001,   'min':0, 'max':0.1},
                'z1':{'value':self.zs, 'min':self.zs - 0.001, 'max':self.zs + 0.001},
                'z2':{'value':self.zs - 0.0002, 'min':self.zs - 0.0005, 'max':self.zs + 0.0005},
                'z3':{'value':self.zs, 'min':self.zs - 0.001, 'max':self.zs + 0.001},   
                'sigma1':{'value':0.0001,  'min':0.0, 'max':0.0005},
                'sigma2':{'value':0.0001,   'min':0.0, 'max':0.0005},
                'sigma3':{'value':0.0001,   'min':0.0, 'max':0.0005},
                'p9':{'value':0.05,   'min':-0.05, 'max':1},
                'p10':{'value':-0.001,  'min':-1, 'max':1 }
            }
        else:
            priors = priors

        # fits = []
        for idx in range(niter):

            # Perturb the spectra 
            ydata = flux_masked + np.array([
                np.random.uniform(low=-i, high=i) for i in err_masked
            ])*1

            try:

                # Set up lmfit Parameters
                params = Parameters()
                params.add('p0', value=priors['p0']['value'], min=priors['p0']['min'], max=priors['p0']['max'])
                params.add('p1', value=priors['p1']['value'], min=priors['p1']['min'], max=priors['p1']['max'])
                params.add('p2', value=priors['p2']['value'], min=priors['p2']['min'], max=priors['p2']['max'])
                params.add('p3', value=priors['p3']['value'], min=priors['p3']['min'], max=priors['p3']['max'])
                params.add('p4', value=priors['p4']['value'], min=priors['p4']['min'], max=priors['p4']['max'])
                params.add('p5', value=priors['p5']['value'], min=priors['p5']['min'], max=priors['p5']['max'])
                
                params.add('z1', value=priors['z1']['value'], min=priors['z1']['min'], max=priors['z1']['max'])
                params.add('z2', value=priors['z2']['value'], min=priors['z2']['min'], max=priors['z2']['max'])
                params.add('z3', value=priors['z3']['value'], min=priors['z3']['min'], max=priors['z3']['max'])
                
                params.add('sigma1', value=priors['sigma1']['value'], min=priors['sigma1']['min'], max=priors['sigma1']['max'])
                params.add('sigma2', value=priors['sigma2']['value'], min=priors['sigma2']['min'], max=priors['sigma2']['max'])
                params.add('sigma3', value=priors['sigma3']['value'], min=priors['sigma3']['min'], max=priors['sigma3']['max'])

                params.add('p9',  value=priors['p9']['value'], min=priors['p9']['min'], max=priors['p9']['max'])
                params.add('p10', value=priors['p10']['value'], min=priors['p10']['min'], max=priors['p10']['max'])

                # x and y data
                xdata = wave_masked * 1e-4
                ydata = ydata
                
                # Fit
                model = Model(stel1296_complex_model)
                result = model.fit(
                    ydata,
                    params,
                    x=xdata,
                    method = method
                )

                # Best-fit values
                # print(result.params.pretty_print())
                
                zstellar.append(result.params['z1'].value)
                zism.append(result.params['z2'].value)
                sigma_stellar.append(result.params['sigma1'].value)
                sigma_ism.append(result.params['sigma2'].value)
                sigma_fineem.append(result.params['sigma3'].value)
                zfineem.append(result.params['z3'].value)
                
                p0.append(result.params['p0'].value)
                p1.append(result.params['p1'].value)
                p2.append(result.params['p2'].value)
                p3.append(result.params['p3'].value)
                p4.append(result.params['p4'].value)
                p5.append(result.params['p5'].value)
                p9.append(result.params['p9'].value)
                p10.append(result.params['p10'].value)

                # fits.append( stel1296_complex_model(wave_masked* 1e-4, *[result.params[name].value for name in result.params]))
                if(plotting):
                    plt.plot(
                        wave_masked,
                        stel1296_complex_model(wave_masked* 1e-4, *[result.params[name].value for name in result.params]),
                        lw=0.5,
                        alpha = 0.3,
                        color='cornflowerblue',
                        drawstyle='steps-mid'
                    )


            except RuntimeError:
                print("run failed")
                
                
        zstellar_med, zstellar_std       = plotHistogram(zstellar, plotting=False)
        zism_med, zism_std               = plotHistogram(zism, plotting=False)
        zfineem_med, zfineem_std         = plotHistogram(zfineem, plotting=False)

        sigma_stellar, sigma_stellar_std = plotHistogram(sigma_stellar, plotting=False)
        sigma_ism, sigma_ism_std         = plotHistogram(sigma_ism, plotting=False)
        sigma_fineem, sigma_fineem_std   = plotHistogram(sigma_fineem, plotting=False)
        
        ## Amplitudes
        p0_med, p0_std                   = plotHistogram(p0, plotting=False)
        p1_med, p1_std                   = plotHistogram(p1, plotting=False)
        p2_med, p2_std                   = plotHistogram(p2, plotting=False)
        p3_med, p3_std                   = plotHistogram(p3, plotting=False)
        p4_med, p4_std                   = plotHistogram(p4, plotting=False)
        p5_med, p5_std                   = plotHistogram(p5, plotting=False)
        p9_med, p9_std                   = plotHistogram(p9, plotting=False)
        p10_med, p10_std                 = plotHistogram(p10, plotting=False)
        
        
        ## Plot the best-fit model
        if(plotting):
            plt.plot(
                wave_masked,
                stel1296_complex_model(wave_masked* 1e-4, p0_med, p1_med, p2_med, p3_med, p4_med, p5_med, zstellar_med, zism_med, zfineem_med, sigma_stellar, sigma_ism, sigma_fineem, p9_med, p10_med),
                lw=3,
                c='red',
                drawstyle='steps-mid'
            )

        results = {
            'zstars':[zstellar_med, zstellar_std],
            'zism':[zism_med, zism_std],
            'zfineem':[zfineem_med, zfineem_std],

            'sigma_stellar':[sigma_stellar, sigma_stellar_std],
            'sigma_ism':[sigma_ism, sigma_ism_std],
            'sigma_fineem':[sigma_fineem, sigma_fineem_std],
            
            'p0':[p0_med, p0_std],
            'p1':[p1_med, p1_std],
            'p2':[p2_med, p2_std],
            'p3':[p3_med, p3_std],
            'p4':[p4_med, p4_std],
            'p5':[p5_med, p5_std],
            'p9':[p9_med, p9_std],
            'p10':[p10_med, p10_std],

            'wave':wave_masked,
            'flux':flux_masked,
            'err':err_masked,
            'fit': stel1296_complex_model(wave_masked* 1e-4, p0_med, p1_med, p2_med, p3_med, p4_med, p5_med, zstellar_med, zism_med, zfineem_med, sigma_stellar, sigma_ism, sigma_fineem, p9_med, p10_med),

        }
        return results
