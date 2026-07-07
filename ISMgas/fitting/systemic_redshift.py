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

    def stellar1300(self, waveMin=1285, waveMax=1315, niter=250, plotting=False):
        
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


        # fits = []
        for idx in range(niter):

            # Perturb the spectra 
            ydata = flux_masked + np.array([
                np.random.uniform(low=-i, high=i) for i in err_masked
            ])*1

            try:

                # Set up lmfit Parameters
                params = Parameters()
                params.add('p0', value=-0.00001, min=-0.04, max=0)
                params.add('p1', value=-0.0001, min=-0.04, max=0)
                params.add('p2', value=-0.00001, min=-0.04, max=0)
                params.add('p3', value=-0.00001, min=-0.05,   max=0)
                params.add('p4', value=-0.001,  min=-0.05,   max=0)
                params.add('p5', value=0.001,   min=0, max=0.1)
                
                params.add('z1', value=self.zs, min=self.zs - 0.001, max=self.zs + 0.001)
                params.add('z2', value=self.zs - 0.0002, min=self.zs - 0.002, max=self.zs + 0.002)
                params.add('z3', value=self.zs, min=self.zs - 0.001, max=self.zs + 0.001)
                
                params.add('sigma1', value=0.00015,  min=0.0, max=0.0005)
                params.add('sigma2', value=0.0003,   min=0.0, max=0.0005)
                params.add('sigma3', value=0.00015,   min=0.0, max=0.0005)

                params.add('p9',  value=0.05,   min=-0.05, max=1)
                params.add('p10', value=-0.001, min=-1, max=1)
                
                # x and y data
                xdata = wave_masked * 1e-4
                ydata = ydata
                
                # Fit
                model = Model(stel1296_complex_model)
                result = model.fit(
                    ydata,
                    params,
                    x=xdata,
                )

                # Best-fit values
                # print(result.params.pretty_print())
                
                zstellar.append(result.params['z1'].value)
                zism.append(result.params['z2'].value)
                sigma_stellar.append(result.params['sigma1'].value)
                sigma_ism.append(result.params['sigma2'].value)
                sigma_fineem.append(result.params['sigma3'].value)
                zfineem.append(result.params['z3'].value)

                # fits.append( stel1296_complex_model(wave_masked* 1e-4, *[result.params[name].value for name in result.params]))
                if(plotting):
                    plt.plot(
                        wave_masked,
                        stel1296_complex_model(wave_masked* 1e-4, *[result.params[name].value for name in result.params]),
                        lw=0.5,
                        alpha = 0.3,
                        c='r',
                        # drawstyle='steps-mid'
                    )


            except RuntimeError:
                print("run failed")
                
                
        zstellar_med, zstellar_std       = plotHistogram(zstellar, plotting=False)
        zism_med, zism_std               = plotHistogram(zism, plotting=False)
        zfineem_med, zfineem_std         = plotHistogram(zfineem, plotting=False)

        sigma_stellar, sigma_stellar_std = plotHistogram(sigma_stellar, plotting=False)
        sigma_ism, sigma_ism_std         = plotHistogram(sigma_ism, plotting=False)
        sigma_fineem, sigma_fineem_std   = plotHistogram(sigma_fineem, plotting=False)

        results = {
            'zstars':[zstellar_med, zstellar_std],
            'zism':[zism_med, zism_std],
            'zfineem':[zfineem_med, zfineem_std],

            'sigma_stellar':[sigma_stellar, sigma_stellar_std],
            'sigma_ism':[sigma_ism, sigma_ism_std],
            'sigma_fineem':[sigma_fineem, sigma_fineem_std],

            'wave':wave_masked,
            'flux':flux_masked,
            'err':err_masked
        }
        return results
