import os
import pickle

import numpy as np
import pandas as pd
import glob

import scipy.stats as stat
from scipy import interpolate
import scipy.interpolate as interp
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
from astropy import units

import astroscrappy
import json

#########################
## Astronomy utilities ##
#########################
def convertDegreesToHMS(ra_deg:float ,dec_deg:float)->str:
    '''
    returns ra and dec in hms from degrees using astropy
    '''
    c = SkyCoord(
        ra    = ra_deg*u.degree,
        dec   = dec_deg*u.degree
    )
    return(c.to_string('hmsdms').replace('h',':').replace('d',':').replace('m',':').replace('s',''))


def saveSpectra(wave, flux, error, fileName, folderPrefix = '', format = 'fits'):
    return(save_spectra(wave, flux, error, fileName, folderPrefix, format))

def save_spectra(wave, flux, error, fileName, folderPrefix = '', format = 'fits'):
    """
    Utility function to save spectra as a fits file (legacy function). 
    Use saveSpectra instead.

    Args:
        wave                            : Wavelength
        flux                            : Flux
        error                           : Error
        fileName                        : filename to save as fits
        folderPrefix (str, optional)    : folder prefix. Defaults to ''.
    """
    if(format == 'fits'):
        t             = Table()
        t['LAMBDA']   = [wave]
        t['SPEC']     = [flux]
        t['ERR']      = [error]
        t['IVAR']     = [1/error**2] ## 1/sigma^^2 = ivar
        
        t.write(folderPrefix+"%s.fits"%(fileName),overwrite=True)

        print("Written file to " + folderPrefix+"%s.fits"%(fileName))
        
        
    if(format=="ascii"):
        t = Table([wave,flux,error], names=('LAMBDA', 'SPEC', 'ERR'))
        t.write(folderPrefix+"%s.txt"%(fileName), format='ascii', overwrite=True)
        

def removeCosmicRays(data, sigclip=2, objlim=2, readnoise=4, verbose=True):
    """
    Returns cosmic ray cleaned data and mask

    Args:
        data, 
        sigclip (int, optional)   : Defaults to 2.
        objlim (int, optional)    : Defaults to 2.
        readnoise (int, optional) : Defaults to 4.
        verbose (bool, optional)  : Defaults to True.
    """
    mask, clean_data = astroscrappy.detect_cosmics(
        data, 
        sigclip   = sigclip,
        objlim    = objlim,
        readnoise = readnoise,
        verbose   = verbose
    )    
    
    return(clean_data, mask)

def makeMedianFolder(folder,objid,crRayClean=False):
    "Make median for a bunch of files in a folder"
    filenames = glob.glob(folder+"*.fits")
    data = []
    for i in filenames:
        if(crRayClean):
            dataClean,_ = removeCosmicRays(fits.getdata(i))
            data.append(dataClean)
        else:
            data.append(fits.getdata(i))
        
    data=np.array(data)
    data_median = np.median(data,axis=0)

    hdr = fits.getheader(filenames[0])
    hdu = fits.PrimaryHDU(data_median.astype(np.float32),header=hdr)
    hdu.writeto(folder+objid+"_median.fits",overwrite=True)


############
### Math ###
############
def interpolateData(x, y, xnew, **kwargs):
    '''
    Assuming y = f(x), returns f(xnew) using scipy's interpolate method.
    '''
    
    fInter  = interpolate.interp1d(x, y,**kwargs)
    return fInter(xnew)


def interpolate2D_scatter(x,y,z, xinterp, yinterp, delta_x, delta_y, units='kpc'):
    """
    Interpolate a set of (x,y,z) scatter points onto (xinterp, yinterp).
    Uses scipy's CloughTocher2DInterpolator.

    Args:
        x (_type_): _description_
        y (_type_): _description_
        z (_type_): _description_
        xinterp (_type_): _description_
        yinterp (_type_): _description_
        delta_x (_type_): _description_
        delta_y (_type_): _description_
        units (str, optional): _description_. Defaults to 'kpc'.
    """
    interpolator = interp.CloughTocher2DInterpolator(
        np.array([x,y]).T, 
        z, 
        fill_value=np.nan, 
        maxiter=1000
    )

    xgrid,ygrid = np.meshgrid(xinterp, yinterp)

    z_interp = interpolator(xgrid, ygrid)
    
    
    ## Store results as a HDU ##
    hdu = fits.PrimaryHDU()
    hdu.data = z_interp
    
    hdu.header['CRVAL1'] = xinterp[0]
    hdu.header['CRVAL2'] = yinterp[0]
    hdu.header['CUNIT1'] = units
    hdu.header['CUNIT2'] = units
    
    hdu.header['CD1_1'] = delta_x
    hdu.header['CD1_2'] = 0    
    hdu.header['CD2_1'] = 0       
    hdu.header['CD2_2'] = delta_y

    return([[xgrid, ygrid, z_interp], hdu])

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


def printStats(qty):
    """Prints some basic statistics of a quantity array

    Args:
        qty (np.array): Quantity array
    """
    print("----------")
    print("min: ", np.min(qty))
    print("max: ", np.max(qty))
    limits = plotHistogram(qty, plotting = False)

    print("Median: ", limits[0])
    print("1sigma: ", limits[1])
    print("----------")
    
def chooseAndPropError(x, xerr, y, yerr, n = 1000):
    '''
    This function takes in arrays (x,xerr) and (y,yerr) and computes what is the 
    difference between x and y by randomly sampling the quantities assuming a 
    normal distribution.
    
    Use Case: 
    
    Say x = velocity measured by person 1 and y = velocity measured by person 2. 
    You want to want how the velocities measured by person 1 differ from those of 
    person 2. You can pass (x, xerr, y, yerr) and this function will give you the 
    mean and std how they vary. 
    
    Input:
    
    x, xerr: quantity (1) and its error
    y, yerr: quantity (2) and its error
    
    Output: 
    
    Mean(Diff(x-y)), std(Diff(x-y)) along with the errors.
        
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


def rotatePoints(x,y, xcenter, ycenter, theta):
    """
    Rotates (x,y) by theta (counter clockwise is position) and returns the output
    """
    x = np.asarray(x)
    y = np.asarray(y)
    theta_ra = np.deg2rad(theta)
        
    x00 = -xcenter + np.cos(theta_ra) * (x + xcenter) - np.sin(theta_ra) * (y + ycenter)
    x01 = -ycenter + np.sin(theta_ra) * (x + xcenter) + np.cos(theta_ra) * (y + ycenter)   
        
        
    return(x00, x01)


###############################
### Uncategorized functions ###
###############################

def save_as_pickle(arrayToSave,fileName:str):
    pickle.dump(arrayToSave, open( fileName, "wb" ) )
    print("file saved to: "+fileName)

def load_pickle(fileName:str):
    return(pickle.load( open(fileName, "rb" ) ))


def save_as_JSON(dictToSave, fileName:str):
    with open(fileName, 'w') as fp:
        json.dump(dictToSave, fp)
    print("file saved to: "+fileName)

def load_JSON(fileName:str):
    with open(fileName, 'r') as fp:
        json_dict = json.load(fp)
    return(json_dict)

def image_to_HTML(path:str,width=600):
    return '<img src="'+ path + '" width="'+ str(width)+'" >'

def makeDirectory(folder):
    try: 
        os.makedirs(folder)
        
    except OSError:
        if not os.path.isdir(folder):
            raise
        
def getNestedArrayValue(db, keys):
    '''
    If db ={
        'x':{
            'y':{
                'z':3
            }
        }
    }
    then getNestdArrayValue(db, ['x','y','z']) returns the value 3 
    getNestdArrayValue(db, ['x','y']) returns {'z':3}
    '''
    value = db
    for i in range(len(keys)):
        value = value.get(keys[i])
        
    return(value)
                

###############################
##### Operations on Images ####
###############################

def image_array(filename:str):
    return(mpimg.imread(filename))

def display_image(filename:str, origin='lower'):
    return(plt.imshow(mpimg.imread(filename), origin=origin))

###########################
########## Plots ###############
###########################

def beautifyPlot(kwargs):
    '''
    Run at end of matplotlib plotting routine.
    '''
    pltFunctions = {
        "acorr":plt.acorr, 
        "angle_spectrum":plt.angle_spectrum, 
        "annotate":plt.annotate, 
        "arrow":plt.arrow, 
        "autoscale":plt.autoscale, 
        "autumn":plt.autumn, 
        "axes":plt.axes, 
        "axhline":plt.axhline, 
        "axhspan":plt.axhspan, 
        "axis":plt.axis, 
        "axline":plt.axline, 
        "axvline":plt.axvline, 
        "axvspan":plt.axvspan, 
        "bar":plt.bar, 
        "bar_label":plt.bar_label, 
        "barbs":plt.barbs, 
        "barh":plt.barh, 
        "bone":plt.bone, 
        "box":plt.box, 
        "boxplot":plt.boxplot, 
        "broken_barh":plt.broken_barh, 
        "cbook":plt.cbook, 
        "cla":plt.cla, 
        "clabel":plt.clabel, 
        "clf":plt.clf, 
        "clim":plt.clim, 
        "close":plt.close, 
        "cm":plt.cm, 
        "cohere":plt.cohere, 
        "color_sequences":plt.color_sequences, 
        "colorbar":plt.colorbar, 
        "colormaps":plt.colormaps, 
        "connect":plt.connect, 
        "contour":plt.contour, 
        "contourf":plt.contourf, 
        "cool":plt.cool, 
        "copper":plt.copper, 
        "csd":plt.csd, 
        "cycler":plt.cycler, 
        "delaxes":plt.delaxes, 
        "disconnect":plt.disconnect, 
        "draw":plt.draw, 
        "draw_all":plt.draw_all, 
        "draw_if_interactive":plt.draw_if_interactive, 
        "errorbar":plt.errorbar, 
        "eventplot":plt.eventplot, 
        "figaspect":plt.figaspect, 
        "figimage":plt.figimage, 
        "figlegend":plt.figlegend, 
        "fignum_exists":plt.fignum_exists, 
        "figtext":plt.figtext, 
        "figure":plt.figure, 
        "fill":plt.fill, 
        "fill_between":plt.fill_between, 
        "fill_betweenx":plt.fill_betweenx, 
        "findobj":plt.findobj, 
        "flag":plt.flag, 
        "functools":plt.functools, 
        "gca":plt.gca, 
        "gcf":plt.gcf, 
        "gci":plt.gci, 
        "get":plt.get, 
        "get_backend":plt.get_backend, 
        "get_cmap":plt.get_cmap, 
        "get_current_fig_manager":plt.get_current_fig_manager, 
        "get_figlabels":plt.get_figlabels, 
        "get_fignums":plt.get_fignums, 
        "get_plot_commands":plt.get_plot_commands, 
        "get_scale_names":plt.get_scale_names, 
        "getp":plt.getp, 
        "ginput":plt.ginput, 
        "gray":plt.gray, 
        "grid":plt.grid, 
        "hexbin":plt.hexbin, 
        "hist":plt.hist, 
        "hist2d":plt.hist2d, 
        "hlines":plt.hlines, 
        "hot":plt.hot, 
        "hsv":plt.hsv, 
        "importlib":plt.importlib, 
        "imread":plt.imread, 
        "imsave":plt.imsave, 
        "imshow":plt.imshow, 
        "inferno":plt.inferno, 
        "inspect":plt.inspect, 
        "install_repl_displayhook":plt.install_repl_displayhook, 
        "interactive":plt.interactive, 
        "ioff":plt.ioff, 
        "ion":plt.ion, 
        "isinteractive":plt.isinteractive, 
        "jet":plt.jet, 
        "legend":plt.legend, 
        "locator_params":plt.locator_params, 
        "logging":plt.logging, 
        "loglog":plt.loglog, 
        "magma":plt.magma, 
        "magnitude_spectrum":plt.magnitude_spectrum, 
        "margins":plt.margins, 
        "matplotlib":plt.matplotlib, 
        "matshow":plt.matshow, 
        "minorticks_off":plt.minorticks_off, 
        "minorticks_on":plt.minorticks_on, 
        "mlab":plt.mlab, 
        "new_figure_manager":plt.new_figure_manager, 
        "nipy_spectral":plt.nipy_spectral, 
        "np":plt.np, 
        "pause":plt.pause, 
        "pcolor":plt.pcolor, 
        "pcolormesh":plt.pcolormesh, 
        "phase_spectrum":plt.phase_spectrum, 
        "pie":plt.pie, 
        "pink":plt.pink, 
        "plasma":plt.plasma, 
        "plot":plt.plot, 
        "plot_date":plt.plot_date, 
        "polar":plt.polar, 
        "prism":plt.prism, 
        "psd":plt.psd, 
        "quiver":plt.quiver, 
        "quiverkey":plt.quiverkey, 
        "rc":plt.rc, 
        "rcParams":plt.rcParams, 
        "rcParamsDefault":plt.rcParamsDefault, 
        "rcParamsOrig":plt.rcParamsOrig, 
        "rc_context":plt.rc_context, 
        "rcdefaults":plt.rcdefaults, 
        "rcsetup":plt.rcsetup, 
        "rgrids":plt.rgrids, 
        "savefig":plt.savefig, 
        "sca":plt.sca, 
        "scatter":plt.scatter, 
        "sci":plt.sci, 
        "semilogx":plt.semilogx, 
        "semilogy":plt.semilogy, 
        "set_cmap":plt.set_cmap, 
        "set_loglevel":plt.set_loglevel, 
        "setp":plt.setp, 
        "show":plt.show, 
        "specgram":plt.specgram, 
        "spring":plt.spring, 
        "spy":plt.spy, 
        "stackplot":plt.stackplot, 
        "stairs":plt.stairs, 
        "stem":plt.stem, 
        "step":plt.step, 
        "streamplot":plt.streamplot, 
        "style":plt.style, 
        "subplot":plt.subplot, 
        "subplot2grid":plt.subplot2grid, 
        "subplot_mosaic":plt.subplot_mosaic, 
        "subplot_tool":plt.subplot_tool, 
        "subplots":plt.subplots, 
        "subplots_adjust":plt.subplots_adjust, 
        "summer":plt.summer, 
        "suptitle":plt.suptitle, 
        "switch_backend":plt.switch_backend, 
        "sys":plt.sys, 
        "table":plt.table, 
        "text":plt.text, 
        "thetagrids":plt.thetagrids, 
        "threading":plt.threading, 
        "tick_params":plt.tick_params, 
        "ticklabel_format":plt.ticklabel_format, 
        "tight_layout":plt.tight_layout, 
        "time":plt.time, 
        "title":plt.title, 
        "tricontour":plt.tricontour, 
        "tricontourf":plt.tricontourf, 
        "tripcolor":plt.tripcolor, 
        "triplot":plt.triplot, 
        "twinx":plt.twinx, 
        "twiny":plt.twiny, 
        "uninstall_repl_displayhook":plt.uninstall_repl_displayhook, 
        "violinplot":plt.violinplot, 
        "viridis":plt.viridis, 
        "vlines":plt.vlines, 
        "waitforbuttonpress":plt.waitforbuttonpress, 
        "winter":plt.winter, 
        "xcorr":plt.xcorr, 
        "xkcd":plt.xkcd, 
        "xlabel":plt.xlabel, 
        "xlim":plt.xlim, 
        "xscale":plt.xscale, 
        "xticks":plt.xticks, 
        "ylabel":plt.ylabel, 
        "ylim":plt.ylim, 
        "yscale":plt.yscale, 
        "yticks":plt.yticks, 

        ## shorthand 
        "l":plt.legend,         
        "t_p":plt.tick_params, 
        "t_l":plt.tight_layout, 
        "t":plt.title, 
        "xlab":plt.xlabel, 
        "xl":plt.xlim, 
        "xs":plt.xscale, 
        "xt":plt.xticks, 
        "ylab":plt.ylabel, 
        "yl":plt.ylim, 
        "ys":plt.yscale, 
        "yt":plt.yticks,         
    }

    for key, arguments in kwargs.items():
        function = pltFunctions.get(key, False)
        if function:
            if(isinstance(arguments,dict)):
                function(**arguments)
                
            if(isinstance(arguments,list)):
                function(*arguments)
        else:
            print(f"Function {key} is not supported!")

def createSubplotGrid(nrows, ncols, figsize=(5,5), dpi=100, **kwargs):
    # Create a figure
    fig = plt.figure(figsize=figsize, dpi=dpi)
    
    # Create a GridSpec layout
    gs = gridspec.GridSpec(nrows, ncols, figure=fig, **kwargs)
    
    # Return the GridSpec object
    return(gs,fig)


def plotWithError(x, y, yerr, sigmaLimit = 1, label = 'data', **kwargs):
    "By default plots x,y and 1 sigma error region"
    plt.plot(
        x,
        y,
        alpha       = kwargs.get('alpha', 0.8),
        linewidth   = kwargs.get('linewidth', 3),
        label       = label,
        color       = kwargs.get('color', color_pal_kvgc['pub1'][16]),
        linestyle   = kwargs.get('linestyle', '-')
    )   

    plt.fill_between(
        x,
        y - yerr * sigmaLimit,
        y + yerr * sigmaLimit,
        alpha       = kwargs.get('sigmaAlpha', color_pal_kvgc['pub1'][7]),
        facecolor   = kwargs.get('facecolor', color_pal_kvgc['pub1'][7])
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

def drawRectange(
    x,y,
    deltax,deltay,
    linewidth=1,edgecolor='r',facecolor='none'):
    """
    Create a matplotlib rectangle patch

    Args:
        x                           : x
        y                           : y
        deltax                      : length in x
        deltay                      : length in y
        linewidth (int, optional)   : Defaults to 1.
        edgecolor (str, optional)   : Defaults to 'r'.
        facecolor (str, optional)   : Defaults to 'none'.
    """
    rect = patches.Rectangle((x, y), deltax,deltay,linewidth=linewidth, edgecolor=edgecolor, facecolor=facecolor)
    return(rect)

def cbarLegendPatch(color, label, **kwargs):
    """
    Returns a cbar patch.
    
    Usage:
    red_patch = cbarLegendPatch('red', 'The red data')
    plt.legend(handles=[red_patch])

    Args:
        color (_type_): _description_
        label (_type_): _description_
    """
    return(patches.Patch(color=color, label=label, **kwargs))

def cbarFontsize(fontsize):
    "Adjust matplotlib colorbar fontsize"
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=fontsize)

def discreteCMap(cmap, n):
    "Returns a discrete colormap with n colors"
    return(plt.get_cmap(cmap,n))


def runCMD(cmd):
    os.system(cmd)
###########################

################################
#### kvgc only use functions ####
################################
# from cdfutils.cdfutils.coords import matrix_to_rot, cdmatrix_to_rscale   
from ISMgas.globalVars import *
from ISMgas.linelist import linelist_highz


def airtovac(wave):
    """ Convert air-based wavelengths to vacuum.
    This code is from pypeit.core.wave.py. https://pypeit.readthedocs.io/en/release/_modules/pypeit/core/wave.html#airtovac

    Parameters
    ----------
    wave: `astropy.units.Quantity`_
        Wavelengths to convert

    Returns
    -------
    new_wave: `astropy.units.Quantity`_
        Wavelength array corrected to vacuum wavelengths
    """
    # Convert to AA
    wave = wave.to(units.AA)
    wavelength = wave.value

    # Standard conversion format
    sigma_sq = (1.e4/wavelength)**2. #wavenumber squared
    factor = 1 + (5.792105e-2/(238.0185-sigma_sq)) + (1.67918e-3/(57.362-sigma_sq))
    factor = factor*(wavelength>=2000.) + 1.*(wavelength<2000.) #only modify above 2000A

    # Convert
    wavelength = wavelength*factor
    # Units
    new_wave = wavelength*units.AA
    new_wave.to(wave.unit)

    return new_wave

def vactoair(wave):
    """Convert to air-based wavelengths from vacuum
    This code is from pypeit.core.wave.py. https://pypeit.readthedocs.io/en/release/_modules/pypeit/core/wave.html#airtovac

    Parameters
    ----------
    wave: `astropy.units.Quantity`_
        Wavelengths to convert

    Returns
    -------
    new_wave: `astropy.units.Quantity`_
        Wavelength array corrected to air

    """
    # Convert to AA
    wave = wave.to(units.AA)
    wavelength = wave.value

    # Standard conversion format
    sigma_sq = (1.e4/wavelength)**2. #wavenumber squared
    factor = 1 + (5.792105e-2/(238.0185-sigma_sq)) + (1.67918e-3/(57.362-sigma_sq))
    factor = factor*(wavelength>=2000.) + 1.*(wavelength<2000.) #only modify above 2000A

    # Convert
    wavelength = wavelength/factor
    new_wave = wavelength*units.AA
    new_wave.to(wave.unit)

    return new_wave


def overlayAllLines(zfactor =1, lowion=True, highion=True, opthin=True, stellar=True, nebem=True, fineem=True, scaleAlpha = 1):
    ## zfactor  = (1+zs)/(1+zinterving) -- zfactor = 1 for zinterving = zs
    
    for i in linelist_highz.keys():
        if(linelist_highz[i].get('lowion') is not None and lowion==True):
            plt.axvline(zfactor*linelist_highz[i]['lambda'], color = 'blue', alpha = scaleAlpha*0.7)

        if(linelist_highz[i].get('highion') is not None and highion==True):
            plt.axvline(zfactor*linelist_highz[i]['lambda'], color = 'purple', alpha = scaleAlpha*0.7)

        if(linelist_highz[i].get('opthin') is not None and opthin==True):
            plt.axvline(zfactor*linelist_highz[i]['lambda'], color = 'yellowgreen', alpha = scaleAlpha*0.7)        

        if(linelist_highz[i].get('stellar') is not None and stellar==True):
            plt.axvline(zfactor*linelist_highz[i]['lambda'], color = 'orange', linestyle= '--', alpha = scaleAlpha*0.7)

        if(linelist_highz[i].get('nebem') is not None and nebem==True):
            plt.axvline(zfactor*linelist_highz[i]['lambda'], color = 'cornflowerblue' ,linestyle='dotted', alpha = scaleAlpha*0.7)        
            
        if(linelist_highz[i].get('fineem') is not None and fineem==True):
            plt.axvline(zfactor*linelist_highz[i]['lambda'], color = 'cornflowerblue' ,linestyle='--', alpha = scaleAlpha*0.9)        


def plotUVSpectra(xdata, ydata, ydataerr, xlim, ylim =[0,2]):
    "Usage : plotUVSpectra(STACK_WAV, STACK_FLUX, 0.05*np.ones(np.shape(STACK_FLUX)), [1200,1350])"
    
    plt.figure(figsize=(12,5))
    plotWithError(
        xdata, 
        ydata,
        ydataerr,
        linewidth = 1.5
    )

    overlayAllLines()

    plt.xlim(xlim)
    plt.ylim(ylim)
    
def bin_ndarray(ndarray, new_shape, operation='sum'):
    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.

    Number of output dimensions must match number of input dimensions and
        new axes must divide old ones.

    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)

    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]

    Taken from stackexchange [ https://stackoverflow.com/questions/8090229/resize-with-averaging-or-rebin-a-numpy-2d-array]

    """
    operation = operation.lower()
    if not operation in ['sum', 'mean']:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d,c in zip(new_shape,
                                                  ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1*(i+1))
    return ndarray


def splitDataCube(filename, nchannels):
    """
    Split a datacube into nchannels. 
    Typically used to split DECALS image into 3 channels.

    Args:
        filename (_type_): _description_
        nchannels (_type_): _description_
    """
    data        = fits.getdata(filename)
    hdr         = fits.getheader(filename)
    wcshst      = WCS(hdr)
    wcshst_2d   = wcshst.dropaxis(2)
    
    for i in range(nchannels):
        ## make dummy hdu and header
        hduFoo        = fits.PrimaryHDU()
        hduFoo.data   = data[i]
        
        hdrFoo        = hduFoo.header
        hdrFoo.update(wcshst_2d.to_header())
        
        hduFoo.writeto(filename.split('.fits')[0] + "_" + str(i) +".fits", overwrite=True)
    

def processCDMatrix(fileName):
    '''
    returns a dict containing pa and pixel scale given an input fits file
    uses cdfutils.
    '''
    
    data_hdr = fits.getheader(filename=fileName)
    print(data_hdr['CD1_1'])
    print(data_hdr['CD1_2'])
    print(data_hdr['CD2_1'])
    print(data_hdr['CD2_2'])

    PA_measured = matrix_to_rot(
        matrix = np.matrix([
            [data_hdr['CD1_1'],data_hdr['CD1_2']],
            [data_hdr['CD2_1'],data_hdr['CD2_2']]
        ])
    )

    print("Measured PA using cdfutils: ", PA_measured)

    pixscale_measured = cdmatrix_to_rscale(
        cdmatrix = np.matrix([
            [data_hdr['CD1_1'],data_hdr['CD1_2']],
            [data_hdr['CD2_1'],data_hdr['CD2_2']]
        ])
    )
    print("Pixel scale is :", pixscale_measured)    
    
    return({'PA': PA_measured, 'pixscale': pixscale_measured})

def save_to_MARZ(wav, spec, sigma, filename):
    """Saves (wav,spec, sigma) from a spectra in a format that MARZ recognizes.
    UNDER CONSTRUCTION

    Args:
        wav (np.array): Wavelength
        spec (np.array): Flux
        sigma (np.array): 1 sigma uncertainity
        filename (string): Filename
    """

    hdu  = fits.PrimaryHDU(spec)
    hdr  = hdu.header
    hdr['CUNIT1'] = 'ANGSTROM'
    hdr['CRPIX1'] = 1    
    hdr['CRVAL1'] = wav[0]
    hdr['CDELT1'] = np.diff(wav)[0]
    
   
    hdu_var  = fits.ImageHDU(sigma**2, header=hdr)
    print(hdu_var.header)
    hdu_list = fits.HDUList([hdu, hdu_var])

    hdu_list.writeto(f'{filename}.fits', overwrite=True)
    
def save_to_QFitsview(wav, spec, sigma, filename):
    """Saves (wav,spec, sigma) from a spectra in a format that QFitsview recognizes.
    UNDER CONSTRUCTION

    Args:
        wav (np.array): Wavelength
        spec (np.array): Flux
        sigma (np.array): 1 sigma uncertainity
        filename (string): Filename
    """

    hdu  = fits.PrimaryHDU(spec)
    hdr  = hdu.header
    hdr['CUNIT1'] = 'ANGSTROM'
    hdr['CRPIX1'] = 1    
    hdr['CRVAL1'] = wav[0]
    hdr['CDELT1'] = np.diff(wav)[0]
    hdu.writeto(f'{filename}.fits', overwrite=True)


def generateHTMLReport(objids):
    df = pd.DataFrame()
    for obj_id in objids:
        df1 = pd.DataFrame(
            [
                [
                image_to_HTML('%s.png'%(obj_id)),        
                image_to_HTML('%s_HST_fitscgi.jpg'%(obj_id)),       
                image_to_HTML('%s-smooth-spectra.png'%(obj_id)),             
                image_to_HTML('%s-all-lines-lowions.png'%(obj_id)), 
                image_to_HTML('%s-lines-verticalstack.png'%(obj_id)), 
                
                image_to_HTML('%s-outflow-profile.png'%(obj_id)),             
                image_to_HTML('%s-outflow-profile-deconvoled.png'%(obj_id)),
                image_to_HTML('%s-fit-continuum.png'%(obj_id)),
            
                image_to_HTML('%s-outflow-lines.png'%(obj_id)),                          
                image_to_HTML('%s-residual.png'%(obj_id)),
                
                image_to_HTML('%s-all-lines-highions.png'%(obj_id)),
                image_to_HTML('%s-fineem-lines.png'%(obj_id)),
                image_to_HTML('%s-nebem-lines.png'%(obj_id)),
                image_to_HTML('%s-opthin-lines.png'%(obj_id)),
                image_to_HTML('%s-stellar-lines.png'%(obj_id))

                ]
            ],
                columns = [
                    'DECALS/HSC image',
                    'HST image',
                    'Smoothed spectra',    
                    'ISM lines used (with range)',  
                    'ISM lines used (vertical stack)',  

                    'Outflow profile',
                    'Outflow profile (deconvolved)',
                    'Outflow profile (continuum fit)',
                
                    'ISM lines used',
                    'Residual (fitting)',

                    'High ionization lines',
                    'Fine structure emission lines',
                    'Nebular emission lines',
                    'Optically thin lines',
                    'Stellar absorption lines'
                ]

        )
        df = pd.concat([df, df1], ignore_index=True)                
    pd.set_option('display.max_colwidth', None)
    print("Summary of results saved to Results/Summary.html")
    df.to_html('Summary.html', escape=False)    
    
    
def runningMedian(x,y,nbins):
    """
    Returns running median of y as a function of x

    Args:
        x (numpy.array): x array
        y (numpy.array)): y array
        nbins (int): Number of bins
    """
    
    bins    = np.linspace(x.min(),x.max(), nbins)
    delta   = bins[1]-bins[0]
    idx     = np.digitize(x,bins)
    running_median = [np.median(y[idx==k]) for k in range(nbins)]
    
    return(bins-delta/2,running_median)





###########################


