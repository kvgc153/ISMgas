import matplotlib.pyplot as plt
import numpy as np  
from ISMgas.linelist import linelist_highz
from ISMgas.SupportingFunctions import plotHistogram
from lmfit import Parameters, Model

def stel1296_complex_model(x, p0, p1, p2, p3, p4, z1, z2, sigma1,sigma2, p9, p10):
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

    center = zstar * 0.129889

    y1294 = p0 * np.exp(-0.5 * ((x - zstar * linelist_highz['Si III 1294']['lambda']*1e-4) / sigma1)**2) / (sigma1 * np.sqrt(2 * np.pi))
    y1296 = p1 * np.exp(-0.5 * ((x - zstar * linelist_highz['Si III 1296']['lambda']*1e-4) / sigma1)**2) / (sigma1 * np.sqrt(2 * np.pi))
    y1298 = p2 * np.exp(-0.5 * ((x - zstar * linelist_highz['Si III 1298-9']['lambda']*1e-4) / sigma1)**2) / (sigma1 * np.sqrt(2 * np.pi))

    y1302 = p3 * np.exp(-0.5 * ((x - zo2 * linelist_highz['O I 1302']['lambda']*1e-4) / sigma2)**2) / (sigma2 * np.sqrt(2 * np.pi))
    y1304 = p4 * np.exp(-0.5 * ((x - zo2 * linelist_highz['Si II 1304']['lambda']*1e-4) / sigma2)**2) / (sigma2 * np.sqrt(2 * np.pi))

    poly_cont = -x**3 + p10 * x**2 + p10 * (x - center)  + p9

    return y1294 + y1296 + y1298 + y1302 + y1304 + poly_cont


class Fitter:
    def __init__(self, wave, flux, err, zs, objid):
        self.wave = wave ## Wavelength in Angstroms
        self.flux = flux
        self.err = err
        self.zs = zs ## Guess redshift
        self.objid = objid

    def stellar1300(self, waveMin=1285, waveMax=1315, niter=250):
        
        mask1 = self.wave > waveMin*(1+self.zs)
        mask2 =  self.wave < waveMax*(1+self.zs)
        mask  = mask1 & mask2

        wave_masked = self.wave[mask]
        flux_masked = self.flux[mask]
        err_masked = self.err[mask]

        plt.figure(figsize=(11,5))

        plt.plot(
            wave_masked, 
            flux_masked,
            lw = 2,
            c = 'black',
            drawstyle='steps-mid'
        )
        zstellar = []
        zism = []
        # fits = []
        for idx in range(niter):

            # Perturb the spectra 
            ydata = flux_masked + np.array([
                np.random.uniform(low=-i, high=i) for i in err_masked
            ])*1

            try:

                # Set up lmfit Parameters
                params = Parameters()
                params.add('p0', value=-0.00001, min=-0.0004, max=0)
                params.add('p1', value=-0.0001, min=-0.0004, max=0)
                params.add('p2', value=-0.00001, min=-0.0004, max=0)
                params.add('p3', value=-0.00001, min=-0.05,   max=0)
                params.add('p4', value=-0.001,  min=-0.05,   max=0)
                
                params.add('z1', value=self.zs, min=self.zs - 0.001, max=self.zs + 0.001)
                params.add('z2', value=self.zs - 0.0002, min=self.zs - 0.002, max=self.zs + 0.002)
                
                params.add('sigma1', value=0.00015,  min=0.0, max=0.0005)
                params.add('sigma2', value=0.0003,   min=0.0, max=0.0005)
                
                params.add('p9',  value=0.05,   min=-0.05, max=1)
                params.add('p10', value=-0.001, min=-1, max=1)
                
                # x and y data
                xdata = wave_masked * 1e-4
                ydata = ydata
                yerr = 0.05*flux_masked
                
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

                # fits.append( stel1296_complex_model(wave_masked* 1e-4, *[result.params[name].value for name in result.params]))

                plt.plot(
                    wave_masked,
                    stel1296_complex_model(wave_masked* 1e-4, *[result.params[name].value for name in result.params]),
                    lw=0.5,
                    c='r',
                    # drawstyle='steps-mid'
                )

            except RuntimeError:
                print("run failed")
                
        plt.show()
                
        zstellar_med, zstellar_std = plotHistogram(zstellar)
        print(zstellar_med, zstellar_std)

        plt.show()
        
        zism_med, zism_std = plotHistogram(zism)
        print(zism_med, zism_std)
        plt.show()
        
        results = {
            'zstars':[zstellar_med, zstellar_std],
            'zism':[zism_med, zism_std],
            'wave':wave_masked,
            'flux':flux_masked,
            'err':err_masked
        }
        return results
