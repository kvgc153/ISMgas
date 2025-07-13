Module ISMgas.SupportingFunctions
=================================

Functions
---------

`airtovac(wave)`
:   Convert air-based wavelengths to vacuum.
    This code is from pypeit.core.wave.py. https://pypeit.readthedocs.io/en/release/_modules/pypeit/core/wave.html#airtovac
    
    Parameters
    ----------
    wave: `astropy.units.Quantity`_
        Wavelengths to convert
    
    Returns
    -------
    new_wave: `astropy.units.Quantity`_
        Wavelength array corrected to vacuum wavelengths

`beautifyPlot(kwargs)`
:   Run at end of matplotlib plotting routine.

`bin_ndarray(ndarray, new_shape, operation='sum')`
:   Bins an ndarray in all axes based on the target shape, by summing or
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

`cbarFontsize(fontsize)`
:   Adjust matplotlib colorbar fontsize

`cbarLegendPatch(color, label, **kwargs)`
:   Returns a cbar patch.
    
    Usage:
    red_patch = cbarLegendPatch('red', 'The red data')
    plt.legend(handles=[red_patch])
    
    Args:
        color (_type_): _description_
        label (_type_): _description_

`chooseAndPropError(x, xerr, y, yerr, n=1000)`
:   This function takes in arrays (x,xerr) and (y,yerr) and computes what is the 
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

`convertDegreesToHMS(ra_deg: float, dec_deg: float) ‑> str`
:   returns ra and dec in hms from degrees using astropy

`createSubplotGrid(nrows, ncols, figsize=(5, 5), dpi=100, **kwargs)`
:   

`discreteCMap(cmap, n)`
:   Returns a discrete colormap with n colors

`display_image(filename: str, origin='lower')`
:   

`drawRectange(x, y, deltax, deltay, linewidth=1, edgecolor='r', facecolor='none')`
:   Create a matplotlib rectangle patch
    
    Args:
        x                           : x
        y                           : y
        deltax                      : length in x
        deltay                      : length in y
        linewidth (int, optional)   : Defaults to 1.
        edgecolor (str, optional)   : Defaults to 'r'.
        facecolor (str, optional)   : Defaults to 'none'.

`find_nearest(input_array, value: float)`
:   

`generateHTMLReport(objids)`
:   

`getNestedArrayValue(db, keys)`
:   If db ={
        'x':{
            'y':{
                'z':3
            }
        }
    }
    then getNestdArrayValue(db, ['x','y','z']) returns the value 3 
    getNestdArrayValue(db, ['x','y']) returns {'z':3}

`image_array(filename: str)`
:   

`image_to_HTML(path: str, width=600)`
:   

`interpolate2D_scatter(x, y, z, xinterp, yinterp, delta_x, delta_y, units='kpc')`
:   Interpolate a set of (x,y,z) scatter points onto (xinterp, yinterp).
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

`interpolateData(x, y, xnew, **kwargs)`
:   Assuming y = f(x), returns f(xnew) using scipy's interpolate method.

`load_JSON(fileName: str)`
:   

`load_QFitsview_spectra(filename, plot=True)`
:   Returns wav and spec from a qfitsview spectra file
    
    Args:
        filename (_type_): _description_

`load_pickle(fileName: str)`
:   

`makeDirectory(folder)`
:   

`makeMedianFolder(folder, objid, crRayClean=False)`
:   Make median for a bunch of files in a folder

`medianMAD(array)`
:   

`overlayAllLines(zfactor=1, lowion=True, highion=True, opthin=True, stellar=True, nebem=True, fineem=True, scaleAlpha=1)`
:   

`plotHistogram(arrayToPlot, arrayName='', bins='scott', method=2, best_fit_plot=True, plotting=True)`
:   Returns 
    method = 1 -- (mu,sigma) using scipy gaussnorm -- great for gaussian like distributions
    method = 2 -- (Median,MAD) using median and MAD -- great for non-uniform distributions
    for a given array

`plotUVSpectra(xdata, ydata, ydataerr, xlim, ylim=[0, 2])`
:   Usage : plotUVSpectra(STACK_WAV, STACK_FLUX, 0.05*np.ones(np.shape(STACK_FLUX)), [1200,1350])

`plotWithError(x, y, yerr, sigmaLimit=1, label='data', **kwargs)`
:   By default plots x,y and 1 sigma error region

`printLinelist(z=0, linelist='highz')`
:   Given a redshift, this function prints the linelist at that redshift
    
    Args:
        z (int, optional): _description_. Defaults to 0.
        linelist (str, optional): _description_. Defaults to 'highz'.

`printStats(qty)`
:   Prints some basic statistics of a quantity array
    
    Args:
        qty (np.array): Quantity array

`processCDMatrix(fileName)`
:   returns a dict containing pa and pixel scale given an input fits file
    uses cdfutils.

`removeCosmicRays(data, inbkg, sigclip=2, objlim=2, readnoise=4, cleantype='medmask', niter=4, verbose=True)`
:   Returns cosmic ray cleaned data and mask
    
    Args:
        data, 
        inbkg : A pre-determined background image, to be subtracted from indat before running the main detection algorithm. This is used primarily with spectroscopic data, to remove sky lines and the cross-section of an object continuum during iteration, “protecting” them from spurious rejection
        sigclip (int, optional)   : Defaults to 2.
        objlim (int, optional)    : Defaults to 2.
        readnoise (int, optional) : Defaults to 4.
        verbose (bool, optional)  : Defaults to True.

`returnWeightedArray(Nsm, spec, ivar, wave_rest)`
:   #### Logic
    
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

`rotatePoints(x, y, xcenter, ycenter, theta)`
:   Rotates (x,y) by theta (counter clockwise is position) and returns the output

`runCMD(cmd)`
:   

`runningMedian(x, y, nbins)`
:   Returns running median of y as a function of x
    
    Args:
        x (numpy.array): x array
        y (numpy.array)): y array
        nbins (int): Number of bins

`saveSpectra(wave, flux, error, fileName, folderPrefix='', format='fits')`
:   

`save_as_JSON(dictToSave, fileName: str)`
:   

`save_as_pickle(arrayToSave, fileName: str)`
:   

`save_spectra(wave, flux, error, fileName, folderPrefix='', format='fits')`
:   Utility function to save spectra as a fits file (legacy function). 
    Use saveSpectra instead.
    
    Args:
        wave                            : Wavelength
        flux                            : Flux
        error                           : Error
        fileName                        : filename to save as fits
        folderPrefix (str, optional)    : folder prefix. Defaults to ''.

`save_to_MARZ(wav, spec, sigma, filename)`
:   Saves (wav,spec, sigma) from a spectra in a format that MARZ recognizes.
    UNDER CONSTRUCTION
    
    Args:
        wav (np.array): Wavelength
        spec (np.array): Flux
        sigma (np.array): 1 sigma uncertainity
        filename (string): Filename

`save_to_QFitsview(wav, spec, sigma, filename)`
:   Saves (wav,spec, sigma) from a spectra in a format that QFitsview recognizes.
    UNDER CONSTRUCTION
    
    Args:
        wav (np.array): Wavelength
        spec (np.array): Flux
        sigma (np.array): 1 sigma uncertainity
        filename (string): Filename

`scipyMuSigma(array)`
:   

`splitDataCube(filename, nchannels)`
:   Split a datacube into nchannels. 
    Typically used to split DECALS image into 3 channels.
    
    Args:
        filename (_type_): _description_
        nchannels (_type_): _description_

`vactoair(wave)`
:   Convert to air-based wavelengths from vacuum
    This code is from pypeit.core.wave.py. https://pypeit.readthedocs.io/en/release/_modules/pypeit/core/wave.html#airtovac
    
    Parameters
    ----------
    wave: `astropy.units.Quantity`_
        Wavelengths to convert
    
    Returns
    -------
    new_wave: `astropy.units.Quantity`_
        Wavelength array corrected to air

`wavelengthToVelocity(WLarray, lambda0)`
: