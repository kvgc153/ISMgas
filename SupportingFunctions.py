
import os
import pickle

import numpy as np
import pandas as pd

import scipy.stats as stat
from scipy import interpolate
from scipy.constants import c, pi
c_kms = c*1e-3

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import gridspec
import matplotlib.patches as patches

from astropy.io import fits
from astropy.visualization import hist 
from astropy.cosmology import FlatLambdaCDM
from astropy.wcs import WCS
from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import Table

#########################
## Astronomy utilities ##
#########################
def convertDegreesToHMS(ra_deg:float ,dec_deg:float)->str:
    '''
    returns ra and dec in hms from degrees using astropy
    '''
    c = SkyCoord(ra=ra_deg*u.degree, dec=dec_deg*u.degree)
    return(c.to_string('hmsdms').replace('h',':').replace('d',':').replace('m',':').replace('s',''))


def save_spectra(wave, flux, error, fileName, folderPrefix = ''):

    t             = Table()
    t['LAMBDA']   = [wave]
    t['SPEC']     = [flux]
    t['ERR']      = [error]
    t['IVAR']     = [1/error**2] ## 1/sigma^^2 = ivar
    
    t.write(folderPrefix+"%s.fits"%(fileName),overwrite=True)

    print("Written file to " + folderPrefix+"%s.fits"%(fileName))

############
### Math ###
############
def interpolateData(x, y, xnew):
    '''
    Assuming y = f(x), returns f(xnew) using scipy's interpolate method.
    '''
    
    fInter  = interpolate.interp1d(x, y)
    return fInter(xnew)


def returnWeightedArray(Nsm, spec, ivar, wave_rest):
    '''
    #### Logic

    $$  wavelength = [\lambda_0,\lambda_1,\lambda_2.......\lambda_n] $$

    $$  flux = [f_0,f_1,f_2,......f_n] $$

    $$  ivar= [iv_0,iv_1,iv_2.......iv_n] $$

    $$ f_{iv} = flux*ivar = [f_0 iv_0 , f_1 iv_1, ... f_n iv_n] $$

    $$ f_{weighted} = Convolve(flux* ivar, kernel size)/Convolve(ivar, kernel size)$$

    $$ standard error  = \sqrt{1/\sum /\sigma_i^2}  = \sqrt{1/1\sum ivar_i} =  \sqrt{1/Convolve(ivar,kernel size)}$$

    $$ \lambda_{weighted} = Convolve(wavlength,kernel size)  $$

    This ensures that the $f_{weighted}$, standard error and the $\lambda_{weighted}$ have the same number of elements.

    #### Input
    - Nsm : Kernel size
    - spec : flux array
    - ivar : invariance array (or 1/weights**2 array)
    - wave_rest : rest frame wavelength

    #### Output
    - weighted average flux
    - weighted averate sigma
    - corrected wavelength

    #### Example :
    ```
    kernel_size= 3
    spec = np.array([1,2,3])
    ivar = np.array([10,100,10])
    wave_rest = np.array([1000,2000,3000])

    returnWeightedArray(kernel_size,spec,ivar,wave_rest)

    [array([2.]), array([2000.]), array([0.09128709])]

    ```
    '''
    return([
        np.convolve(spec*ivar, np.ones(Nsm),mode = 'valid')/np.convolve(ivar, np.ones(Nsm),mode = 'valid'),
        np.convolve(wave_rest, np.ones((Nsm,))/Nsm, mode='valid'),
        1/np.sqrt(np.convolve(ivar, np.ones((Nsm)), mode='valid'))
    ])
    
def wavelengthToVelocity(WLarray, lambda0):
    return(c_kms*((np.asarray(WLarray)-lambda0)/lambda0))
    
def find_nearest(input_array, value:float):
    array   = np.asarray(input_array)
    idx     = (np.abs(array - value)).argmin()
    return array[idx]   

def scipyMuSigma(array):
    mu, sigma = stat.norm.fit(array)
    return(mu, sigma)

def medianMAD(array):
    mu    = np.median(array,axis=0)
    MAD   = np.median(np.abs(array-mu))
    ##Approximately Converting MAD to sigma - https://en.wikipedia.org/wiki/Median_absolute_deviation   
    sigma = MAD/0.675
    
    return(mu, sigma)

def chooseAndPropError(x, xerr, y, yerr, n = 1000):
    '''
    This function propagates the errors by randomly sampling the quantities 
    within the errors provided. Assumes a normal distribution for the xerr and yerr quantities.
    
    Input:
    
    x, xerr: DG fit quantity and its error
    y, yerr: SG fit quantity and its error
        
    '''
    assert ((len(x) == len(y)) and (len(x) == len(xerr)) and (len(y) == len(yerr))), "Ensure that length of arrays are same"
    allValues = []
    allStds = []
    for j in range(n):
        foo_std = []
        for i in range(len(x)):
            chosen_x        = np.random.normal(x[i], xerr[i], 1)
            chosen_y        = np.random.normal(y[i], yerr[i], 1)
            quantity_diff   = chosen_x - chosen_y

            foo_std.append(quantity_diff)
            
        allStds.append(np.std(foo_std))
        allValues.append(np.mean(foo_std))

    allValues = np.array(allValues) 
    allStds = np.array(allStds)
    print("Mean:%d $\pm$ %d, St.Dev: %d $\pm$ %d"%(np.mean(allValues), np.std(allValues),np.mean(allStds),np.std(allStds)))
    return([np.mean(allValues), np.std(allValues),np.mean(allStds),np.std(allStds)])

###############################
### Uncategorized functions ###
###############################

def save_as_pickle(arrayToSave,fileName:str):
    pickle.dump(arrayToSave, open( fileName, "wb" ) )
    print("file saved to: "+fileName)

def load_pickle(fileName:str):
    return(pickle.load( open(fileName, "rb" ) ))

def image_to_HTML(path:str,width=600):
    return '<img src="'+ path + '" width="'+ str(width)+'" >'

def makeDirectory(folder):
    try: 
        os.makedirs(folder)
        
    except OSError:
        if not os.path.isdir(folder):
            raise

###############################
##### Operations on Images ####
###############################

def image_array(filename:str):
    return(mpimg.imread(filename))

def display_image(filename:str):
    return(plt.imshow(mpimg.imread(filename)))


###########################
########## Plots ###############
###########################

def beautifyPlot(kwargs):
    '''
    Run at end of matplotlib plotting routine. Thanks to Patrick Wells (UCD) for suggesting a clean way to implement this 
    Example:
    ```
    x   = np.arange(0,100,1)
    y   = x**2

    plt.plot(x,y, label= 'y=f(x)')

    beautifyPlot({
        'title'         : {'label': '$x^2$ v/s x'},
        'xlim'          : {'left': 0, 'right':100},
        'ylim'          : {'bottom': 0, 'top':10000},
        'legend'        : {'bbox_to_anchor' : (1, 1), 'fontsize': 15},
        'tightlayout'   : {}

    })

    ```
    
    '''
    pltFunctions = {
        "title"         : plt.title,
        "xlabel"        : plt.xlabel,
        "ylabel"        : plt.ylabel,
        "xlim"          : plt.xlim,
        "ylim"          : plt.ylim,
        "xticks"        : plt.xticks,
        "yticks"        : plt.yticks,
        "tightlayout"   : plt.tight_layout,
        "legend"        : plt.legend,
        "grid"          : plt.grid,
        "savefig"       : plt.savefig,
        "suptitle"      : plt.suptitle
    }

    for key, arguments in kwargs.items():
        function = pltFunctions.get(key, False)
        if function:
            function(**arguments)
        else:
            print(f"Function {key} is not supported!")


def plotWithError(x, y, yerr, sigmaLimit = 1, label = 'data', **kwargs):
    "By default plots x,y and 1 sigma error region"
    
    plt.plot(
        x,
        y,
        alpha       = 0.8,
        linewidth   = kwargs.get('linewidth', 3),
        label       = label,
        color       = color_pal_kvgc['pub1'][16]
    )

    plt.fill_between(
        x,
        y - yerr * sigmaLimit,
        y + yerr * sigmaLimit,
        alpha       = 0.5,
        facecolor   = color_pal_kvgc['pub1'][7]
    )


def plotHistogram(
        arrayToPlot,
        arrayName       = '',
        bins            = 'scott',
        method          = 2,
        best_fit_plot   = True,
        plotting        = True
    ):
    """
    Returns 
    method = 1 -- (mu,sigma) using scipy gaussnorm -- great for gaussian like distributions
    method = 2 -- (Median,MAD) using median and MAD -- great for non-uniform distributions
    for a given array
    """
    arrayToPlot = np.array(arrayToPlot)

    if(method==1):
        ## Uses scipy norm fit -- works great for gaussian like distribution
        
        mu, sigma     = scipyMuSigma(arrayToPlot)
        if(plotting):
            _, bins, _ = hist(
                arrayToPlot,
                color       = 'black',
                linewidth   = 6,
                alpha       = 0.8,
                bins        = bins,
                histtype    = 'step',
                density     = True
            )

            best_fit_line = stat.norm.pdf(bins, mu, sigma)
            plt.plot(bins, best_fit_line)
            plt.axvline(mu,linestyle='-',linewidth=4,color='purple',alpha=0.7)
            plt.axvline(mu+sigma,linestyle='--',linewidth=2.5,color='green',alpha=0.5)
            plt.axvline(mu-sigma,linestyle='--',linewidth=2.5,color='green',alpha=0.5)
            plt.title("$\mu_{fit}$=%.3f $\sigma_{fit}=$%.3f"%(mu,sigma),fontsize=15)

            plt.xticks(fontsize=15,rotation=90)
            plt.yticks(fontsize=15)
            plt.ylabel("Counts (Normalized)",fontsize=20)
            plt.xlabel(arrayName,fontsize=20)
            plt.grid('on',alpha=0.6)

    elif(method==2):

        ##### MAD - Median absolute deviation-- works great for non-gaussian distribution ####
        mu, sigma     = medianMAD(arrayToPlot)

        if(plotting):
            _, bins, _ = hist(
                arrayToPlot,
                color       = 'black',
                linewidth   = 6,
                alpha       = 0.8,
                bins        = bins,
                histtype    = 'step',
                density     = True
            )

            if(best_fit_plot):
                best_fit_line = stat.norm.pdf(bins, mu, sigma)
                plt.plot(bins, best_fit_line)
                plt.axvline(mu,linestyle='-',linewidth=4, color='purple',alpha=0.7)
                plt.axvline(mu+sigma,linestyle='--',linewidth=2.5,color='green',alpha=0.5)
                plt.axvline(mu-sigma,linestyle='--',linewidth=2.5,color='green',alpha=0.5)
                plt.title("Median=%.1f 1$\sigma$=%.2f"%(mu,sigma),fontsize=15)

            plt.xticks(fontsize=15,rotation=90)
            plt.yticks(fontsize=15)
            plt.ylabel("Counts (Normalized)",fontsize=20)
            plt.xlabel(arrayName,fontsize=20)
            plt.grid('on',alpha=0.6)

    return(mu,sigma)

def drawRectange(x,
                 y,
                 deltax,
                 deltay,
                 linewidth=1,
                 edgecolor='r',
                 facecolor='none'):
    rect = patches.Rectangle((x, y), deltax,deltay,linewidth=linewidth, edgecolor=edgecolor, facecolor=facecolor)
    return(rect)
###########################