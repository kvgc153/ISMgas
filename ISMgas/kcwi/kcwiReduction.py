from ISMgas.kcwi.kcwiFunctions import *
from ISMgas.visualization.fits import ScaleImage

from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from astropy.visualization import ZScaleInterval, ImageNormalize, PercentileInterval

class kcwiReduxMontage:
    def __init__(self, galaxy, filenames, pixscale=0.3, **kwargs):
        self.ra = galaxy.ra
        self.dec = galaxy.dec
        self.objid = galaxy.objid
        self.filenames = filenames
        self.pixscale = pixscale ## arcseconds
        self.dataCube  = None
        
              
    def combineAndDrizzle(self, drizzleFactor=0.7, HST=0, plotting=True, **kwargs):
        
        ### Write a DECaLS like header
        template = f"""
SIMPLE  =                    T / file does conform to FITS standard
BITPIX  =                  -32 / number of bits per data pixel
NAXIS   =                    2 / number of data axes
NAXIS1  =                  140 / length of data axis 1
NAXIS2  =                  140 / length of data axis 2
EXTEND  =                    T / FITS dataset may contain extensions
COMMENT   FITS (Flexible Image Transport System) format is defined in 'Astronomy
COMMENT   and Astrophysics', volume 376, page 359; bibcode: 2001A&A...376..359H
SURVEY  = 'LegacySurvey'
VERSION = 'DR9     '
IMAGETYP= 'IMAGE   '           / None
CTYPE1  = 'RA---TAN'           / TANgent plane
CTYPE2  = 'DEC--TAN'           / TANgent plane
CRVAL1  =              {self.ra} / Reference RA
CRVAL2  =             {self.dec} / Reference Dec
CRPIX1  =                 70.5 / Reference x
CRPIX2  =                 70.5 / Reference y
CD1_1   = -{self.pixscale/3600} / CD matrix
CD1_2   =                   0. / CD matrix
CD2_1   =                   0. / CD matrix
CD2_2   = {self.pixscale/3600} / CD matrix
END
"""
        f = open(f"{self.objid}.hdr", 'w+')
        f.write(template)
        f.close()        
        
        dd = kcwiAnalysis(
            filename = self.filenames,
        )

        if(HST==1):
            hdrName = kwargs.get('hdrName', 'SCI')
            fits.writeto(
                filename = self.objid + ".fits",
                data = dd.dataCube,
                header = fits.getheader(self.filenames[0],extname=hdrName),
                overwrite= True
            )          
        else:
            fits.writeto(
                filename = self.objid + ".fits",
                data = dd.dataCube,
                header = fits.getheader(self.filenames[0]),
                overwrite= True
            ) 

        ## Project datacube using montage
        cmd = f"mProjectCube -X -z 0.7 {self.objid}.fits  {self.objid}_drizzle.fits  {self.objid}.hdr"
        runCMD(cmd)
        print(f"Drizzled file: {self.objid}_drizzle.fits")
        
        ## Show user the drizzled datacube
        self.dataCube = fits.getdata(f"{self.objid}_drizzle.fits")
        dd = kcwiAnalysis(
            objid = 'test',
            filename = [f"{self.objid}_drizzle.fits"],
        )
        print(f"Shape: {np.shape(dd.dataCube)}")
        if(plotting):
            ScaleImage(dd.dataCubeMean).plot()
        

def padAndAlign(cubes, newOutputShape, centroids=[],idx=[500,-500],method='mean'):
    
    results = []
    alignStack = []
        
    for i in range(len(cubes)):
        
        newCube = np.zeros(newOutputShape)
        newCube[:,0:cubes[i].dataCube.shape[1], 0:cubes[i].dataCube.shape[2]] = cubes[i].dataCube

        
        if(len(centroids)==0):
            results.append(newCube)
            alignStack.append(np.nanmean(newCube[idx[0]:idx[1], :, :], axis=0))
            
        else:
            newAlignCube = np.roll(
                newCube, 
                [0, -(centroids[i][0] - centroids[0][0]), -(centroids[i][1]- centroids[0][1])],
                axis=(0,2,1)
            )
            results.append(newAlignCube)
            alignStack.append(np.nanmean(newAlignCube[idx[0]:idx[1] :, :], axis=0))
        
    fits.writeto(
        filename = 'align.fits',
        data = np.array(alignStack),
        overwrite= True
    )
    print("Use align.fits to manually align the datacubes")
    if(method=='mean'):
        return(np.mean(results,axis=0))
    
    elif(method=='sum'):
        return(np.sum(results,axis=0))
    elif(method=='individual'):
        return(results)
    
    
    
    
def combineRedSide(filename, skymask, objectmask, output, xlim=[6000,9000], ylim=[-50,1000], hduNum = 4, method='median'):
    print(f"processing {filename}")
    maskSky = fits.getdata(skymask)
    maskObj = fits.getdata(objectmask)
    
    hdu = fits.open(filename)
    data = hdu[hduNum].data  ## Typically 4 is the NOSKYSUB extension
    data_median = np.nanmedian(data,axis=0)

    plt.figure()
    plt.imshow(data_median,origin='lower', norm= ImageNormalize(data_median, PercentileInterval(99)), cmap='gray')
    plt.contour(maskSky, levels=[0.5], colors='orange')
    plt.contour(maskObj, levels=[0.5], colors='purple')
    plt.show()
    plt.close()

    ##############################################
    wavstart = int(hdu[0].header['CRVAL3'])
    wav_air = np.array(range(wavstart,wavstart+len(data),1))
    
    
    obj_spectra = []
    y,x = np.where(maskObj>0)
    for i in range(len(x)):
        obj_spectra.append(data[:,y[i],x[i]])
    
    sky_spectra = []
    y,x = np.where(maskSky>0)
    for i in range(len(x)):
        sky_spectra.append(data[:,y[i],x[i]])
    
    plt.figure(figsize=(12,5), dpi=150)

    if(method=='median'):
        median_sky =  np.median(sky_spectra, axis=0)
        obj = np.median(obj_spectra, axis=0)

    if(method=='mean'):
        median_sky =  np.mean(sky_spectra, axis=0)
        obj = np.mean(obj_spectra, axis=0)

    
    plt.plot(
        wav_air,
        obj,
        color='black',
        drawstyle='steps-mid',
        label = 'object spectra'
    
    )
    
    plt.plot(
        wav_air,
        median_sky,
        color='orange',
        drawstyle='steps-mid',
        label = 'sky model'
    
    )

    
    # plt.axvline([7483.5072])
    plt.ylim(ylim)
    plt.xlim(xlim)

    plt.show()
    plt.close()

    ### Write new datacube #####

    dataNew = np.zeros(np.shape(data))
    yFoo, xFoo = np.where(maskObj >-10)
    
    def gaussian(x, amp, mean, sigma):
        return amp * np.exp(-0.5 * ((x - mean) / sigma)**2)
    def gaussian_with_bias(x, amp, mean, sigma, bias):
        return amp * np.exp(-0.5 * ((x - mean) / sigma)**2) + bias


    for i in range(len(xFoo)):

        objectTrace = data[:, yFoo[i], xFoo[i]] - median_sky
        objRunningMedian = medfilt(objectTrace, kernel_size=31)
        objectTrace -= objRunningMedian
        
        ## But this object trace still contains many "peaks" that we need to remove manually
        
        std_dev = np.std(objectTrace)  # Standard deviation of the data
        mean_val = np.mean(objectTrace)  # Mean of the data
        optimal_height = mean_val + 2 * std_dev
        
        # Find peaks in the sky_spectra
        peaks, _ = find_peaks(objectTrace, height=optimal_height)  
        peak_wavelengths = wav_air[peaks]
    
    
        cleaned_object_spectra = (data[:, yFoo[i], xFoo[i]] - median_sky).copy()
    
        # Loop over each detected peak
        for peak_wave in peak_wavelengths:
            # Define a small window around the peak
            window = (wav_air > peak_wave - 5) & (wav_air < peak_wave + 5)
            xdata = wav_air[window]
            ydata = cleaned_object_spectra[window]
        
            # Initial guesses for Gaussian parameters: amp, mean, sigma
            guess_amp = np.max(ydata) - np.median(ydata)
            guess_mean = peak_wave
            guess_sigma = 0.5
            guess_bias = np.median(ydata)

        
            try:
                # Define bounds for the parameters: (amp, mean, sigma, bias)
                bounds = (
                    [0, peak_wave - 0.1, 0, -np.inf],  # Lower bounds
                    [np.inf, peak_wave + 0.1, 4, np.inf]  # Upper bounds
                )

                # Fit the Gaussian with bias to the data
                popt, _ = curve_fit(
                    gaussian_with_bias, 
                    xdata, 
                    ydata, 
                    p0=[guess_amp, guess_mean, guess_sigma, guess_bias], 
                    bounds=bounds
                )

                # Subtract the fitted Gaussian with bias from the object spectra
                cleaned_object_spectra[window] -= gaussian_with_bias(xdata, *popt)

            
            except RuntimeError:
                # Fitting might fail sometimes; just skip in that case
                continue

            
        dataNew[:, yFoo[i], xFoo[i]] = cleaned_object_spectra
        

    
    # dataNew[:, yFoo, xFoo] = data[:, yFoo, xFoo] - median_sky[:, np.newaxis]



    
    
    hduNew = fits.PrimaryHDU()
    hduNew.header = hdu[0].header
    hduNew.header["COMMENT"] = "Subtracting median sky"
    hduNew.data = dataNew
    hduNew.writeto(output , overwrite=True)
    print(f"written file to {output}")




from reproject import reproject_exact, reproject_interp
from reproject.mosaicking import find_optimal_celestial_wcs
from scipy.signal import correlate2d
from scipy.ndimage import shift

def preprocess(filename, slicer = 'medium'):
    hdu1 = fits.PrimaryHDU()
    WCS1 = WCS(fits.getheader(filename))
    WCS1 = WCS1.dropaxis(2)
    hdr = WCS1.to_fits()[0].header
    
    frame =  fits.getdata(filename)
    if(slicer=='medium'):
        # Pad the edges with zeros, to remove noisy data
        frame[0:17,:] = np.nan
        frame[-18:,:] = np.nan
        
        frame[0:17,:] = np.nan
        frame[-18:,:] = np.nan
        
        frame[:,0:18,:] = np.nan
        frame[:,80:,:] = np.nan
        
    elif(slicer=='large'):
        # Pad the edges with zeros, to remove noisy data
        frame[:,0:14,:] = np.nan
        frame[:,79:,:] = np.nan

        frame[:,:,:2] = np.nan
        frame[:,:,25:] = np.nan
    
    hdu1.data  = np.nanmedian(frame,axis=0)
    hdu1.header = hdr

    return(hdu1)

def preprocessCube(filename, slicer = 'medium'):
    hdu1 = fits.PrimaryHDU()
    WCS1 = WCS(fits.getheader(filename))
    WCS1 = WCS1.dropaxis(2)
    hdr = WCS1.to_fits()[0].header
    
    frame =  fits.getdata(filename)
    if(slicer=='medium'):
        # Pad the edges with zeros, to remove noisy data
        frame[0:17,:] = np.nan
        frame[-18:,:] = np.nan
        
        frame[0:17,:] = np.nan
        frame[-18:,:] = np.nan
        
        frame[:,0:18,:] = np.nan
        frame[:,80:,:] = np.nan
        
    elif(slicer=='large'):
        # Pad the edges with zeros, to remove noisy data
        frame[:,0:14,:] = np.nan
        frame[:,79:,:] = np.nan

        frame[:,:,:2] = np.nan
        frame[:,:,25:] = np.nan
    
    hdu1.data  = frame
    hdu1.header = hdr

    return(hdu1)



def reproject_and_mosaic(hdus, method='exact', apply_shift=True):
    """
    Reproject multiple 2D images onto a common WCS frame, 
    align them using 2D cross-correlation, and combine into a mosaic.

    Parameters
    ----------
    hdus : list of astropy.io.fits.PrimaryHDU or ImageHDU
        List of 2D image HDUs with WCS.
    method : str, optional
        Reprojection method: 'exact' (slow, accurate) or 'interp' (fast, approximate).

    Returns
    -------
    mosaic_data : np.ndarray
        The combined image data.
    mosaic_wcs : astropy.wcs.WCS
        The WCS object of the mosaic.
    """
    if len(hdus) == 0:
        raise ValueError("No HDUs provided.")
    if method not in ['exact', 'interp']:
        raise ValueError("method must be 'exact' or 'interp'.")

    # Choose the reprojection function
    reproj_func = reproject_exact if method == 'exact' else reproject_interp

    # Find optimal WCS
    mosaic_wcs, mosaic_shape = find_optimal_celestial_wcs(hdus)

    # Initialize
    combined_data = np.zeros(mosaic_shape, dtype=float)
    weight_map = np.zeros(mosaic_shape, dtype=float)

    # Choose the first image as reference
    ref_hdu = hdus[0]
    ref_data, _ = reproj_func(ref_hdu, mosaic_wcs, shape_out=mosaic_shape)

    # Add the reference image
    valid = ~np.isnan(ref_data)
    combined_data[valid] += ref_data[valid]
    weight_map[valid] += 1

    for hdu in hdus[1:]:
        # Reproject the target image
        target_data, _ = reproj_func(hdu, mosaic_wcs, shape_out=mosaic_shape)

        # Cross-correlation to find shift
        valid_mask = (~np.isnan(ref_data)) & (~np.isnan(target_data))
        if np.sum(valid_mask) == 0:
            continue  # Skip if no overlap
        
        if(apply_shift):
            corr = correlate2d(
                np.nan_to_num(ref_data) * valid_mask, 
                np.nan_to_num(target_data) * valid_mask,
                mode="same"
            )

            shift_y, shift_x = np.array(np.unravel_index(np.argmax(corr), corr.shape)) - np.array(corr.shape) // 2

            # Apply the shift
            shifted_data = shift(target_data, shift=(shift_y, shift_x), order=1, mode='constant', cval=np.nan)
            
        else:
            shifted_data = target_data

        # Combine
        valid = ~np.isnan(shifted_data)
        combined_data[valid] += shifted_data[valid]
        weight_map[valid] += 1

    # Normalize
    with np.errstate(divide='ignore', invalid='ignore'):
        mosaic_data = np.where(weight_map > 0, combined_data / weight_map, np.nan)

    return mosaic_data, mosaic_wcs


def reproject_and_mosaic_cube(hdus, spectral_axis=0, parallel=1, method='exact', apply_shift=True):
    """
    Reproject multiple data cubes onto a common WCS frame that covers all of them,
    align them spatially using cross-correlation (once), and combine them into a single cube mosaic.

    Parameters
    ----------
    hdus : list of astropy.io.fits.PrimaryHDU or ImageHDU
        List of 3D data cube HDUs with WCS.
    spectral_axis : int, optional
        Axis index corresponding to the spectral dimension (default=0).
    parallel : bool, optional
        Whether to run reproject_exact in parallel (ignored for reproject_interp).
    method : str, optional
        'exact' for reproject_exact or 'interp' for reproject_interp.

    Returns
    -------
    mosaic_cube : np.ndarray
        The combined 3D data cube (spectral, y, x).
    mosaic_wcs : astropy.wcs.WCS
        The WCS object of the mosaic (spatial only; spectral axis is preserved separately).
    """
    if len(hdus) == 0:
        raise ValueError("No HDUs provided.")

    if method not in ['exact', 'interp']:
        raise ValueError("method must be 'exact' or 'interp'.")

    n_spectral = hdus[0].data.shape[spectral_axis]

    # Use median images for spatial alignment
    spatial_hdus = []
    for hdu in hdus:
        median_image = np.nanmedian(hdu.data, axis=spectral_axis)
        spatial_hdus.append(fits.ImageHDU(median_image, header=hdu.header))

    mosaic_wcs, mosaic_shape = find_optimal_celestial_wcs(spatial_hdus)

    # Compute shifts using median images
    print("Computing shifts between datacubes...")
    ref_median = np.nanmedian(hdus[0].data, axis=spectral_axis)
    ref_hdu = fits.ImageHDU(ref_median, header=hdus[0].header)
    
    if method == 'exact':
        ref_data, _ = reproject_exact(ref_hdu, mosaic_wcs, shape_out=mosaic_shape, parallel=parallel)
    else:
        ref_data, _ = reproject_interp(ref_hdu, mosaic_wcs, shape_out=mosaic_shape)

    shifts = [(0.0, 0.0)]

    for hdu in hdus[1:]:
        median_image = np.nanmedian(hdu.data, axis=spectral_axis)
        target_hdu = fits.ImageHDU(median_image, header=hdu.header)

        if method == 'exact':
            target_data, _ = reproject_exact(target_hdu, mosaic_wcs, shape_out=mosaic_shape, parallel=parallel)
        else:
            target_data, _ = reproject_interp(target_hdu, mosaic_wcs, shape_out=mosaic_shape)

        valid_mask = (~np.isnan(ref_data)) & (~np.isnan(target_data))
        if np.sum(valid_mask) == 0:
            shifts.append((0.0, 0.0))
            continue

        if(apply_shift):
            # Perform 2D cross-correlation between the reference and target data
            # Multiply by valid_mask to ensure only valid regions are considered
            corr = correlate2d(
                np.nan_to_num(ref_data) * valid_mask,
                np.nan_to_num(target_data) * valid_mask,
                mode="same"
            )

            # Find the indices of the maximum correlation value
            # Subtract half the correlation shape to calculate the shift
            shift_y, shift_x = (
                np.array(np.unravel_index(np.argmax(corr), corr.shape)) - np.array(corr.shape) // 2
            )
            # Append the calculated shift (shift_y, shift_x) to the shifts list
            shifts.append((shift_y, shift_x))
            
        else:
            # If no shift is to be applied, append a default shift of (0.0, 0.0)
            shifts.append((0.0, 0.0))

    # Initialize output cubes
    mosaic_cube = np.full((n_spectral, *mosaic_shape), np.nan)
    weight_cube = np.zeros((n_spectral, *mosaic_shape), dtype=float)

    # Now apply shifts and reproject full cubes
    for ndatacube, (hdu, (shift_y, shift_x)) in enumerate(zip(hdus, shifts)):
        cube_data = hdu.data

        for i in range(n_spectral):
            print(f"Reprojecting slice {i + 1}/{n_spectral} of datacube-{ndatacube}...")
            if spectral_axis == 0:
                slice_data = cube_data[i, :, :]
            else:
                raise ValueError("Unsupported spectral_axis value. Must be 0.")

            slice_hdu = fits.ImageHDU(slice_data, header=hdu.header)

            if method == 'exact':
                reproj_slice, footprint = reproject_exact(slice_hdu, mosaic_wcs, shape_out=mosaic_shape, parallel=parallel)
            else:
                reproj_slice, footprint = reproject_interp(slice_hdu, mosaic_wcs, shape_out=mosaic_shape)

            shifted_slice = shift(reproj_slice, shift=(shift_y, shift_x), order=1, mode='constant', cval=0)
            shifted_footprint = shift(footprint, shift=(shift_y, shift_x), order=1, mode='constant', cval=0.0)

            valid = shifted_footprint > 0
            mosaic_cube[i][valid] = np.nan_to_num(mosaic_cube[i][valid]) + np.nan_to_num(shifted_slice[valid])
            weight_cube[i][valid] += 1

    # Normalize by weight
    with np.errstate(divide='ignore', invalid='ignore'):
        mosaic_cube = np.where(weight_cube > 0, mosaic_cube / weight_cube, np.nan)

    return mosaic_cube, mosaic_wcs
