import numpy as np
import matplotlib.pyplot as plt
from lmfit import Parameters, Model
from ISMgas.SupportingFunctions import plotHistogram

## Fitting a triple Gaussian where the first Gaussian is fixed at v0=0 and the other two are free to vary. The function returns the best fit parameters and their uncertainties.
def triple_gaussian(v, A1, sigma1, A2, v02, sigma2, A3, v03, sigma3):
    g1 = A1 * np.exp(-0.5 * (v / sigma1)**2)
    g2 = A2 * np.exp(-0.5 * ((v - v02) / sigma2)**2)
    g3 = A3 * np.exp(-0.5 * ((v - v03) / sigma3)**2)
    return 1.0 - g1 - g2 - g3


def fitTripleGaussian(priors, niter, xdata,ydata,ydataerr,method='basinhopping', plotting=True, doubleGaussian=False):  
    fit_results = {
        'A1': [],
        'sigma1': [],
    
        'A2': [],
        'v02': [],
        'sigma2': [],
    
        'A3': [],
        'v03': [],
        'sigma3': []
    }
    if priors is None:
        priors = {
                'A1': {'value': 0.5, 'min': 0.4, 'max': 0.7},
                'sigma1': {'value': 120.0, 'min': 115, 'max': 125},

                'A2': {'value': 0.5, 'min': 0.2, 'max': 1.0},
                'v02': {'value': 100.0, 'min': 0.0, 'max': 100.0},
                'sigma2': {'value': 100.0, 'min': 50, 'max': 1000.0},

                'A3': {'value': 0.5, 'min': 0.2, 'max': 1.0},
                'v03': {'value': -50.0, 'min': -500.0, 'max': 0},
                'sigma3': {'value': 100.0, 'min': 50, 'max': 1000.0}
            }

    
    model = Model(triple_gaussian)
    params = Parameters()
    
    if(doubleGaussian):
        params.add('A1', value=0, vary=False)
        params.add('sigma1', value=1, vary=False)
    else:
        params.add('A1', value=priors['A1']['value'], min=priors['A1']['min'], max=priors['A1']['max'])
        params.add('sigma1', value=priors['sigma1']['value'], min=priors['sigma1']['min'], max=priors['sigma1']['max'])
    
    params.add('A2', value=priors['A2']['value'], min=priors['A2']['min'], max=priors['A2']['max'])
    params.add('v02', value=priors['v02']['value'], min=priors['v02']['min'], max=priors['v02']['max'])
    params.add('sigma2', value=priors['sigma2']['value'], min=priors['sigma2']['min'], max=priors['sigma2']['max'])
    
    params.add('A3', value=priors['A3']['value'], min=priors['A3']['min'], max=priors['A3']['max'])
    params.add('v03', value=priors['v03']['value'], min=priors['v03']['min'], max=priors['v03']['max'])
    params.add('sigma3', value=priors['sigma3']['value'], min=priors['sigma3']['min'], max=priors['sigma3']['max'])
    
    
    import time 
    successful_fits =0
    for i in range(niter):
        np.random.seed(int(time.time())+i)
        if((i+1)%10==0):
            print(f"Iteration {i+1}/{niter}")
        y_perturbed = ydata + np.array([np.random.uniform(low=-i,high=i) for i in ydataerr])
        
        try:
            result_mc = model.fit(
                y_perturbed,
                params,
                v= xdata,
                method = method
            )
    
            # Only keep successful fits
            if result_mc.success:
                successful_fits += 1
    
                for name in fit_results:
                    fit_results[name].append(
                        result_mc.params[name].value
                    )
        except Exception as e:
            print(f"Iteration {i+1} failed: {e}")
    
    print(f"\nSuccessful fits:{successful_fits}/{niter}")
    
    for name in fit_results:
        fit_results[name] = np.array(fit_results[name])
    
    print("\n" + "="*60)
    print("MONTE CARLO RESULTS")
    print("="*60)
    
    best_values = {}
    uncertainties = {}
    
    for name, values in fit_results.items():
        # Median, MAD
        mu,sigma = plotHistogram(values, plotting=False)
    
        best_values[name] = mu
        uncertainties[name] = sigma
    
        print(
            f"{name:7s} = "
            f"{mu:.5g} "
            f"+{sigma:.5g} "
        )
    
    # ============================================================
    # Construct final model using MC median parameters
    # ============================================================
    A1 = best_values['A1']
    sigma1 = best_values['sigma1']
    
    A2 = best_values['A2']
    v02 = best_values['v02']
    sigma2 = best_values['sigma2']
    
    A3 = best_values['A3']
    v03 = best_values['v03']
    sigma3 = best_values['sigma3']
    
    vfit = np.arange(-2000,2000,1)## Sampled at 1km/s for EW calculations
    
    g1 = A1 * np.exp(-0.5 * (vfit / sigma1)**2)
    g2 = A2 * np.exp(-0.5 * ((vfit - v02) / sigma2)**2)
    g3 = A3 * np.exp(-0.5 * ((vfit - v03) / sigma3)**2)
    yfit_mc = 1.0 - g1 - g2 - g3
    
    EW_g1_kms = np.nansum(g1) * 1.0
    EW_g2_kms = np.nansum(g2) * 1.0
    EW_g3_kms = np.nansum(g3) * 1.0
    
    # ============================================================
    # Plot final MC best fit
    # ============================================================
    if plotting:
        plt.figure(figsize=(9, 6))
        
        plt.errorbar(
            xdata,
            ydata,
            yerr=ydataerr,
            fmt='o',
            markersize=4,
            capsize=2,
            color='black',
            alpha=0.7,
            label='Data'
        )
        
        plt.plot(
            vfit,
            yfit_mc,
            'r-',
            linewidth=2.5,
            label='MC best fit'
        )
        
        plt.plot(
            vfit,
            1.0 - g1,
            '--',
            linewidth=1.5,
            label='Gaussian 1: $v_0=0$'
        )
        
        plt.plot(
            vfit,
            1.0 - g2,
            '--',
            linewidth=1.5,
            label='Gaussian 2'
        )
        
        plt.plot(
            vfit,
            1.0 - g3,
            '--',
            linewidth=1.5,
            label='Gaussian 3'
        )
        
        plt.xlabel('Velocity (km/s)')
        plt.ylabel('Y')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        
        # ============================================================
        # Plot the three absorption components correctly
        # ============================================================
        
        plt.figure(figsize=(9, 6))
        
        plt.errorbar(
            xdata,
            ydata,
            yerr=ydataerr,
            fmt='o',
            markersize=4,
            capsize=2,
            color='black',
            alpha=0.6,
            label='Data'
        )
        
        plt.plot(
            vfit,
            yfit_mc,
            'k-',
            linewidth=2.5,
            label='Total fit'
        )
        
        plt.plot(
            vfit,
            1.0 - g1,
            '--',
            label='Gaussian 1: $v_0=0$'
        )
        
        plt.plot(
            vfit,
            1.0 - g2,
            '--',
            label=f'Gaussian 2: $v_0={v02:.1f}$ km/s'
        )
        
        plt.plot(
            vfit,
            1.0 - g3,
            '--',
            label=f'Gaussian 3: $v_0={v03:.1f}$ km/s'
        )
        
        plt.xlabel('Velocity (km/s)')
        plt.ylabel('Y')
        plt.legend()
        plt.tight_layout()
        plt.show()
        # ============================================================
        # Plot parameter distributions
        # ============================================================
        fig, axes = plt.subplots(
            4, 2,
            figsize=(12, 14)
        )
        
        axes = axes.flatten()
        
        for ax, (name, values) in zip(
            axes,
            fit_results.items()
        ):
        
            ax.hist(
                values,
                bins=20,
                color='steelblue',
                alpha=0.75,
                edgecolor='black'
            )
            mu, sigma = plotHistogram(values, plotting=False)
        
            ax.axvline(
                mu,
                color='red',
                linewidth=2,
                label='Median'
            )
        
            ax.axvline(
                mu - sigma,
                color='red',
                linestyle='--'
            )
        
            ax.axvline(
                mu + sigma,
                color='red',
                linestyle='--'
            )
        
            ax.set_xlabel(name)
            ax.set_ylabel('N')
            ax.legend()
        
        plt.tight_layout()
        plt.show()
    return({
        'fit_results': fit_results,
        'best_values': best_values,
        'uncertainties': uncertainties,
        'vfit': vfit,
        'g1': g1,
        'g2': g2,
        'g3': g3,
        'yfit_mc': yfit_mc,
        'EW_g1_kms': EW_g1_kms,
        'EW_g2_kms': EW_g2_kms,
        'EW_g3_kms': EW_g3_kms
    })