import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import ImageNormalize, ZScaleInterval
import matplotlib.pyplot as plt

from reproject import reproject_exact, reproject_interp
from reproject.mosaicking import find_optimal_celestial_wcs

from symfit.core.minimizers import DifferentialEvolution,BFGS,BasinHopping
from symfit import Poly, variables, parameters, Model, Fit, cos,GreaterThan,LessThan

import os
from scipy import ndimage
from scipy.signal import correlate2d
from scipy.ndimage import shift

from ISMgas.visualization.fits import ScaleImage
from ISMgas.GalaxyProperties import GalaxyProperties
from ISMgas.kcwi.kcwiFunctions import kcwiAnalysis
from ISMgas.SupportingFunctions import plotWithError

def preprocess(filename, slicer = 'medium', cube = False):
    """Pad edges of the raw KCWI datacube/image before reducing the data. 

    Args:
        filename (string): Filename of datacube/image to preprocess
        slicer (str, optional): Which KCWI slicer did you use?. Defaults to 'medium'.
        cube (bool, optional): Is the file a datacube or just an image?. Defaults to False.
    """
    frame =  fits.getdata(filename)
    
    if(slicer=='small'):
        frame[:,0:29,:] = np.nan
        frame[:,163:,:] = np.nan

        frame[:,:,:11] = np.nan
        frame[:,:,32:] = np.nan       
    
    if(slicer=='medium'):
        frame[:,0:16,:] = np.nan
        frame[:,79:,:] = np.nan

        frame[:,:,0:5] = np.nan
        frame[:,:,-5:] = np.nan
        
    elif(slicer=='large'):
        frame[:,0:14,:] = np.nan
        frame[:,79:,:] = np.nan

        frame[:,:,:2] = np.nan
        frame[:,:,25:] = np.nan
        
    if(cube):
        hdu1 = fits.PrimaryHDU()   
        hdr   = fits.getheader(filename)
        hdu1.data   = frame
        hdu1.header = hdr
    
    else:
        hdu1 = fits.PrimaryHDU()
        WCS1 = WCS(fits.getheader(filename))
        WCS1 = WCS1.dropaxis(2)
        hdr = WCS1.to_fits()[0].header
        hdu1.data  = np.nanmedian(frame[500:-500,:,:],axis=0)
        hdu1.header = hdr

    return(hdu1)

def kcwi_check_samewave(hdr0, hdr1):
    """
    This code is from kcwikit -- https://github.com/yuguangchen1/KcwiKit/blob/master/kcwikit/kcwi/kcwi.py
    Check if the wavelength axes are the same in two headers.

    Args:
        hdr0 (astropy.io.fits.header) - input header #0
        hdr1 (astropy.io.fits.header) - input header #1

    Returns:
        boolean: whether wave axes are the same
    """

    if hdr0['NAXIS3'] != hdr1['NAXIS3']:
        # Not the same amount of pixels
        return False

    wave0 = (np.arange(hdr0['NAXIS3']) - hdr0['CRPIX3'] + 1) * hdr0['CD3_3'] + hdr0['CRVAL3']
    wave1 = (np.arange(hdr1['NAXIS3']) - hdr1['CRPIX3'] + 1) * hdr1['CD3_3'] + hdr1['CRVAL3']

    if not np.isclose(wave0[0], wave1[0]):
        # Starting point different
        return False
    if not np.isclose(wave0[1] - wave0[0], wave1[1] - wave1[0]):
        # delta w different
        return False

    return True



def kcwi_resample_wave(hdu, newhdr, method='cubic',plot=False):
    """
    This code is from kcwikit -- https://github.com/yuguangchen1/KcwiKit/blob/master/kcwikit/kcwi/kcwi.py

    Resample a cube to match the wavelength direction to a different header.

    Args:
        hdu (astropy HDU): input hdu
        newhdr (astropy header): header containing the new wavelength grid
        order (str): interpolation method or 'mask'

    Returns:
        astropy.io.fits.PrimaryHDU: resampled data cube in the form of HDU
    """
    from scipy import interpolate

    hdr = hdu.header
    wave = (np.arange(hdr['NAXIS3']) - hdr['CRPIX3'] + 1) * hdr['CD3_3'] + hdr['CRVAL3']
    data = hdu.data.copy()
    #print("oldwave with length %d" % hdr['NAXIS3'],"starting at %.2f" % hdr['CRVAL3'], " delta %.4f" % hdr['CD3_3'])
    #print("newwave with length %d" % newhdr['NAXIS3'],"starting at %.2f" % newhdr['CRVAL3'], " delta %.4f" % newhdr['CD3_3'])
    newwave = (np.arange(newhdr['NAXIS3']) - newhdr['CRPIX3'] + 1) * newhdr['CD3_3'] + newhdr['CRVAL3']
    newdata = np.zeros((len(newwave), hdu.shape[1], hdu.shape[2])) + np.nan
    #print("newdata with shape", newdata.shape)
    #print("olddata with shape", data.shape)

    data = data.reshape(len(wave), -1)
    newdata = newdata.reshape(len(newwave), -1)
    #print(newdata.shape)
    for i in range(newdata.shape[1]):
        spec = data[:, i]

        if method != 'mask':
            mask = ~np.isfinite(spec)
            spec = np.nan_to_num(spec)

            # No good data, skip
            if np.sum(spec)==0:
                continue

            ci = interpolate.interp1d(wave, spec, kind=method, bounds_error=False, fill_value=np.nan)
            #print(wave)
            #print(newwave)
            
            newspec = ci(newwave)

            mi = interpolate.interp1d(wave, mask, kind='linear', bounds_error=False, fill_value=1)
            newmask = mi(newwave)

            newspec[newmask != 0] = np.nan

        else:
            # mask cube
            mask = spec

            mi = interpolate.interp1d(wave, mask, kind='linear', bounds_error=False, fill_value=128)
            newmask = mi(newwave)

            newspec = newmask

        newdata[:, i] = newspec

    newdata = newdata.reshape((len(newwave), hdu.shape[1], hdu.shape[2]))

    if plot:
        plt.plot(wave,hdu.data[:,68,15],"g")
        plt.plot(newwave,newdata[:,68,15],"r")
    newhdu = hdu.copy()
    newhdu.header['NAXIS3'] = newhdr['NAXIS3']
    newhdu.header['CRPIX3'] = newhdr['CRPIX3']
    newhdu.header['CRVAL3'] = newhdr['CRVAL3']
    newhdu.header['CD3_3'] = newhdr['CD3_3']
    newhdu.data = newdata

    return newhdu


def _process_single_datacube(
    hdu,
    mosaic_wcs,
    mosaic_shape,
    spectral_axis,
    method,
    shift_xy,  # pixels (y, x), applied AFTER reprojection
    cube_index,
    outdir,
):

    os.makedirs(outdir, exist_ok=True)

    shift_y_pix, shift_x_pix = shift_xy
    cube_data = hdu.data
    n_spectral = cube_data.shape[spectral_axis]

    # 2D WCS
    wcs2d = WCS(hdu.header).dropaxis(2)
    header2d = wcs2d.to_fits()[0].header

    processed_cube = np.full((n_spectral, *mosaic_shape), np.nan)
    weight_cube = np.zeros((n_spectral, *mosaic_shape), dtype=float)

    for i in range(n_spectral):
        if i % 250 == 0:
            print(f"Datacube-{cube_index}: Processing {i}")

        slice_data = cube_data[i, :, :]
        slice_hdu = fits.ImageHDU(slice_data, header=header2d)

        if method == 'exact':
            reproj, footprint = reproject_exact(
                slice_hdu, mosaic_wcs, shape_out=mosaic_shape
            )
        else:
            reproj, footprint = reproject_interp(
                slice_hdu, mosaic_wcs, shape_out=mosaic_shape
            )

        # Apply pixel shift AFTER reprojection
        reproj_shifted = ndimage.shift(
            reproj,
            shift=(shift_y_pix, shift_x_pix),
            order=1,        # bilinear (same spirit as interp)
            mode='constant',
            cval=np.nan
        )

        footprint_shifted = ndimage.shift(
            footprint,
            shift=(shift_y_pix, shift_x_pix),
            order=0,        # nearest-neighbor for mask
            mode='constant',
            cval=0.0
        )

        valid = footprint_shifted > 0
        processed_cube[i][valid] = np.nan_to_num(reproj_shifted[valid])
        weight_cube[i][valid] += 1

    processed_cube[weight_cube > 0] /= weight_cube[weight_cube > 0]
    processed_cube[weight_cube == 0] = np.nan

    outfile = os.path.join(outdir, f"processed_cube_{cube_index}.fits")
    print(f"Done processing {outfile}")
    fits.writeto(outfile, processed_cube, overwrite=True)

    return outfile

def reproject_and_mosaic(
    hdus,
    method='exact',
    search_size=25,
    offset = (0,0),
    autocorrelate=False,
    autocorrelate_maskfile=None,
    correlate_mode='full',
    resolution=None
):

    if len(hdus) == 0:
        raise ValueError("No HDUs provided.")
    if method not in ['exact', 'interp']:
        raise ValueError("method must be 'exact' or 'interp'.")

    if autocorrelate_maskfile is not None:
        userMask = fits.getdata(autocorrelate_maskfile).astype(float)
        userMask[userMask == 0] = np.nan
    else:
        userMask = None

    reproj_func = reproject_exact if method == 'exact' else reproject_interp

    if resolution is None:
        mosaic_wcs, mosaic_shape = find_optimal_celestial_wcs(hdus)
    else:
        mosaic_wcs, mosaic_shape = find_optimal_celestial_wcs(hdus, resolution=resolution)
        
    print("Optimal WCS determined. Starting reprojection and mosaic...")

    # Reference frame
    ref_hdu = hdus[0]
    ref_data, _ = reproj_func(ref_hdu, mosaic_wcs, shape_out=mosaic_shape)
    ref_data = np.nan_to_num(ref_data)

    print("Reference frame reprojected. Starting autocorrelation and alignment...")

    shifted_frames = [ref_data]
    shifts = [(0.0, 0.0)]

    # The following code uses similar approach as KCWIKit to find the optimal shifts
    # KCWI-style parameters
    search_size = search_size
    conv_filter = 2
    upfactor = 10
    offset = offset    
    print(f"Autocorrelation parameters: search_size={search_size}, conv_filter={conv_filter}, upfactor={upfactor}, offset={offset}")


    for hdu in hdus[1:]:
        tgt_data, _ = reproj_func(hdu, mosaic_wcs, shape_out=mosaic_shape)
        tgt_data = np.nan_to_num(tgt_data)

        if not autocorrelate:
            shifted_frames.append(tgt_data)
            shifts.append((0.0, 0.0))
            continue

        # ---------- COARSE SEARCH ----------
        crls_size = search_size + conv_filter
        xx = np.arange(-crls_size, crls_size + 1) + offset[1]
        yy = np.arange(-crls_size, crls_size + 1) + offset[0]
        crls = np.zeros((len(xx), len(yy)))

        for i, dx in enumerate(xx):
            for j, dy in enumerate(yy):
                shifted = shift(tgt_data, (dx, dy), order=1, mode='constant', cval=0.0)
                if userMask is not None:
                    valid = (userMask == userMask) # Convert to boolean
                    mult = ref_data[valid] * shifted[valid]
                else:
                    mult = ref_data * shifted
                if np.any(mult):
                    crls[i, j] = np.sum(mult)

        # Apply a maximum filter to find local maxima in the correlation map
        # The filter size is (2*conv_filter+1) to match the convolution kernel size
        max_conv = ndimage.maximum_filter(crls, 2 * conv_filter + 1)
        
        # Create a boolean mask identifying pixels that are local maxima and have non-zero correlation
        maxima = (crls == max_conv) & (crls != 0)
        labeled, _ = ndimage.label(maxima)
        slices = ndimage.find_objects(labeled)

        dxs, dys = [], []
        for slc in slices:
            cx = (slc[0].start + slc[0].stop - 1) // 2
            cy = (slc[1].start + slc[1].stop - 1) // 2
            dxs.append(cx)
            dys.append(cy)

        dxs = np.array(dxs)
        dys = np.array(dys)
        r = xx[dxs] ** 2 + yy[dys] ** 2
        idx = np.argmin(r)

        shift_y = xx[dxs[idx]]
        shift_x = yy[dys[idx]]
        
        print(f"Coarse shift found: (y: {shift_y}, x: {shift_x})")
        print("Starting fine search...")

        # ---------- FINE SEARCH ----------
        ref_up = ndimage.zoom(ref_data, upfactor, order=1, grid_mode=True, mode='grid-constant')
        tgt_up = ndimage.zoom(tgt_data, upfactor, order=1, grid_mode=True, mode='grid-constant')

        ncrl = upfactor
        fx = np.arange(-ncrl, ncrl + 1)
        fy = np.arange(-ncrl, ncrl + 1)
        crls_fine = np.zeros((len(fx), len(fy)))

        for i, dx in enumerate(fx):
            for j, dy in enumerate(fy):
                shifted = shift(
                    tgt_up,
                    (shift_y * upfactor + dx, shift_x * upfactor + dy),
                    order=1,
                    mode='constant',
                    cval=0.0
                )
                mult = ref_up * shifted
                if np.any(mult):
                    crls_fine[i, j] = np.sum(mult)

        mi, mj = np.unravel_index(np.argmax(crls_fine), crls_fine.shape)
        shift_y += fx[mi] / upfactor
        shift_x += fy[mj] / upfactor
        
        print(f"Fine shift found: (y: {shift_y}, x: {shift_x})")

        shifted_data = shift(
            tgt_data,
            (shift_y, shift_x),
            order=1,
            mode='constant',
            cval=np.nan
        )
        # print(shift_x,shift_y, resolution, resolution.value)

        shifted_frames.append(shifted_data)
        shifts.append((shift_y, shift_x))

    return shifted_frames, shifts, mosaic_wcs, mosaic_shape


def reproject_and_mosaic_cube(
    hdus,
    mosaic_wcs,
    mosaic_shape,
    objid,
    spectral_axis=0,
    method='exact',
    shifts=[]
):
    from joblib import Parallel, delayed
    import numpy as np
    from astropy.io import fits

    ## Begin Checks ## 
    if len(hdus) == 0:
        raise ValueError("No HDUs provided.")

    if len(hdus) != len(shifts):
        raise ValueError("shifts length must match number of HDUs")
    
    ## Check if all cubes have same wavelength axis
    changeWavelength = False # FLag
    for i in range(1, len(hdus)):
        if not kcwi_check_samewave(hdus[0].header, hdus[i].header):
            # raise ValueError(f"The wavelength axes of the {0} and {i} cubes are not the same. Fix this before proceeding.")
            print(f"The wavelength axes of the {0} and {i} cubes are not the same")
            changeWavelength = True
    
    newhdus = [hdus[0]]
    if(changeWavelength):
        print("Resampling all wavelengths to the first frame")
        for i in range(1, len(hdus)):
            print(f"Resampling {i+1} datacube...")
            fooHdu = kcwi_resample_wave(hdus[i], hdus[0].header)
            newhdus.append(fooHdu) # Resample hdus[i+1] to hdus[0] header
        del hdus 
        hdus = newhdus
    ## End checks ## 

    processed_files = Parallel(n_jobs=-1)(
        delayed(_process_single_datacube)(
            hdu=hdus[i],
            mosaic_wcs=mosaic_wcs,
            mosaic_shape=mosaic_shape,
            spectral_axis=spectral_axis,
            method=method,
            shift_xy=shifts[i],
            cube_index=i,
            outdir=f"{objid}_processed_cubes",
        )
        for i in range(len(hdus))
    )

    # --- combine step (unchanged interface) ---
    cubes = [fits.getdata(f) for f in processed_files]
    mosaic_cube = np.zeros_like(cubes[0])
    weight_cube = np.zeros_like(cubes[0])
    shifted_frames = []

    for cube in cubes:
        valid = np.isfinite(cube)
        shifted_frames.append(np.nanmedian(cube,axis=0))
        mosaic_cube[valid] += cube[valid]
        weight_cube[valid] += 1
        
    shiftedframe_hdu         = fits.PrimaryHDU()
    shiftedframe_hdu.data    = shifted_frames
    shiftedframe_hdu.header  = mosaic_wcs.to_fits()[0].header


    mosaic_cube[weight_cube > 0] /= weight_cube[weight_cube > 0]
    mosaic_cube[weight_cube == 0] = np.nan

    return mosaic_cube, shiftedframe_hdu


def getSkyModel(flux, mask, plotting=False, verbose=False):
    """Given a 2D flux map and a mask, this function will return a sky model map using a 2D first-order polynomial

    Args:
        flux (numpy array): Flux
        mask (numpy array): Mask 
        plotting (bool, optional): Plots the before and after sky subtraction. Defaults to False.
        verbose (bool, optional): Prints the fit result. Defaults to False.
    """
    x, y, z                = variables('x, y, z')
    c0,c1, c2,c3,c4,c5,c6  = parameters('c0,c1,c2,c3,c4,c5,c6')
    
    # Make a polynomial. Note the `as_expr` to make it symfit friendly.
    model_dict = {
        z: Poly( {(0, 0): c0,(1, 0): c1, (0, 1): c2,(1, 1): c3}, x ,y).as_expr()
    }
    model = Model(model_dict)

    obj_flux = mask*flux
    
    yd,xd           = np.where(~np.isnan(obj_flux))
    zdata           = obj_flux[yd,xd]  # Removed the values from the edges of the datacube which are padded in preprocess() from fitting
    mask_nonzero    = flux!=0

    # Perform the fit
    fit           = Fit(model, x=yd, y=xd, z=zdata)
    fit_result    = fit.execute()

    zfit = model(x=yd, y=xd, **fit_result.params)
    if verbose:
        print(fit_result)

    sky_model           = np.zeros(obj_flux.shape)
    sky_model[yd,xd]    = zfit
    sky_model           = sky_model * mask_nonzero
    
    if(plotting):
        plt.figure(figsize = (15,7), dpi= 400)
        plt.subplot(1,3,1)
        plt.imshow(flux*mask,origin='lower',cmap = 'RdBu')
        plt.title("Sky from data")
        plt.colorbar()
        
        plt.subplot(1,3,2)
        plt.imshow(sky_model,origin='lower',cmap = 'RdBu')
        plt.title("Model of the sky - First order fit")
        plt.colorbar()

        plt.subplot(1,3,3)
        x,y = np.where(flux*mask!=0)
        z = (flux*mask)[x,y]
        plt.hist(z,label='Before sky correction',alpha=0.6)

        z = ((flux-sky_model)*mask)[x,y]
        plt.hist(z,label='After sky correction',alpha = 0.6)
        plt.legend()
        
        plt.tight_layout()

    # I don't remember why we have to redo this
    yd,xd               = np.where(~np.isnan(obj_flux))
    zfit                = model(x=yd, y=xd, **fit_result.params)
    sky_model           = np.zeros(obj_flux.shape)
    sky_model[yd,xd]    = zfit
    sky_model           = sky_model * mask_nonzero
    
    return(sky_model)

def makeErrorSpectra(maskFile, hdrFile, xIdx, yIdx, xlim=[3300,5000], ylim=[0,0.02]):
    """Used to make an error spectra for a fully reduced KCWI datacube. 
    (need better documentation)

    Args:
        maskFile (_type_): _description_
        hdrFile (_type_): _description_
        xIdx (_type_): _description_
        yIdx (_type_): _description_
        xlim (list, optional): _description_. Defaults to [3300,5000].
        ylim (list, optional): _description_. Defaults to [0,0.02].

    Returns:
        _type_: _description_
    """
    mask    = fits.getdata(maskFile)
    hdr     = fits.getheader(hdrFile)
    
    
    dd = kcwiAnalysis(
        filename = hdrFile,
    )
    plt.figure(dpi=100)
    plt.imshow(dd.dataCubeMean, origin='lower', norm= ImageNormalize(dd.dataCubeMean, interval=ZScaleInterval()))
    plt.imshow(mask, origin='lower', alpha = 0.4)
    plt.show()

    y,x           = np.where(mask==1)
    sky_values    = []
    for i in range(len(x)):
        sky_values.append(dd.dataCube[:, y[i], x[i]])
        
    sky_values = np.asarray(sky_values)
    
    MEAN    = np.median(sky_values, axis=0)
    STD     = np.std(sky_values, axis=0)
    # VAR = 1/STD**2

    plt.figure(dpi=250)
    plotWithError(    
        (np.arange(hdr['NAXIS3']) - hdr['CRPIX3'] + 1) * hdr['CDELT3'] + hdr['CRVAL3'],
        MEAN, 
        STD
    )
    plt.axhline([0], color='black')
    plt.xlim(xlim)
    plt.ylim([-5*np.nanmedian(STD,axis=0), 5*np.nanmedian(STD,axis=0)])
    plt.show()


    plt.figure(dpi=250)
    plt.plot(    
        (np.arange(hdr['NAXIS3']) - hdr['CRPIX3'] + 1) * hdr['CDELT3'] + hdr['CRVAL3'],
        STD
    )
    plt.axhline([0], color='black')
    plt.xlim(xlim)
    plt.ylim(0,5*np.nanmedian(STD,axis=0))
    plt.ylabel("$\sigma$")
    plt.xlabel("$\lambda(\AA)$")
    plt.show()


    plt.figure(dpi=250)
    plt.plot(
        (np.arange(hdr['NAXIS3']) - hdr['CRPIX3'] + 1) * hdr['CDELT3'] + hdr['CRVAL3'],
        dd.dataCube[:, yIdx, xIdx],
        color = 'black',
        linewidth = 0.3
    )
    
    plt.fill_between(
        x =        (np.arange(hdr['NAXIS3']) - hdr['CRPIX3'] + 1) * hdr['CDELT3'] + hdr['CRVAL3'],
        y1 = dd.dataCube[:,yIdx, xIdx] - STD, 
        y2 =  dd.dataCube[ :,yIdx, xIdx] + STD,
        color = 'gray', 
        alpha = 0.4
    )
    
    plt.ylim(ylim)
    plt.xlim(xlim)

    return STD

class kcwiRedux:
    def __init__(self, 
                 objid, ra, dec, 
                 resolution, size, 
                 filenames, slicer, 
                 search_size = 25,
                 offset = (0,0),
                 grab=False, layer='ls-dr9',
                 autocorrelate=True, autocorrelate_maskfile = None, correlate_mode='full', 
                 offsets_arcsecond = [],
                 skymaskFilenames = None):
        """
        Args:
            objid (string): Unique object ID
            ra (float): _RA
            dec (float): DEC
            resolution (arcseconds): Resolution of final datacube in arcseconds
            size (_type_): _description_
            filenames (_type_): _description_
            slicer (_type_): _description_
            grab (bool, optional): _description_. Defaults to False.
            autocorrelate (bool, optional): _description_. Defaults to True.
            autocorrelate_maskfile (_type_, optional): _description_. Defaults to None.
            correlate_mode (str, optional): _description_. Defaults to 'full'.
            skymaskFilenames (_type_, optional): _description_. Defaults to None.
        """
        self.filenames          = filenames
        self.skymaskFilenames   = skymaskFilenames
        self.objid              = objid
        self.slicer             = slicer
        self.resolution         = resolution
        self.search_size        = search_size
        self.offset             = offset
        self.offsets_arcsecond  = offsets_arcsecond

        
        gg = GalaxyProperties(
            ra= ra,
            dec = dec,
            objid = objid + "_DECALS",
          
        )
        gg.decalsFitsAndPng(grab=grab, pixscale = resolution.value, size=size,   layer = layer)

        ## Autocorrelate and auto align all the datcubes 
        self.autocorrelate            = autocorrelate
        self.autocorrelate_maskfile   = autocorrelate_maskfile
        self.correlate_mode           = correlate_mode
        
             
        ## Show user color image of all the datacubes that are going to be reduced 
        self.showWhiteLightImage()
        
    def showWhiteLightImage(self):
        """
        Shows the user a white light image of all the datacubes that are going to be reduced. 
        This is just for the user to check that the datacubes look correct before reducing.
        """
        nrow = 4 
        ncol = len(self.filenames)//nrow + 1 
        
        plt.figure(figsize=(nrow*4, ncol*6), dpi = 200)
        for count,name in enumerate(self.filenames):
            plt.subplot(ncol,nrow,count+1)
            ScaleImage(
                np.nanmedian(fits.getdata(name)[500:-500,:,:], axis=0),
                cmap = 'gray'
            ).plot()
    
            plt.title(name.split("/")[-1], fontsize=10)
            plt.axis('off')
            plt.tight_layout()
        
        plt.savefig(self.objid+"_inputDatacubes.png")
        plt.show()
        

    def step1(self, hduRef=None, refCombine='mean'):
        """
        Step 1 of the reduction process.
        This step will align all the datacubes to a common WCS and save the shifts to a file.
        """
        
        hdus = []
        if(hduRef is None):
            ## If the user has not provided a reference image, we will use the DECaLS image as the reference image for alignment.
            ## By default we will use the mean of the DECaLS image 
            ## But if for some reason that fails (too bright targets in the red). then the user can pass an integer to specify the band 
            ## 0 - g band, 1 - r band, 2 - z band
            hduRef              = fits.open( self.objid + "_DECALS.fits")
            hduRef[0].header    = WCS(hduRef[0].header).dropaxis(2).to_fits()[0].header
            
            if(refCombine=='mean'):
                hduRef[0].data = np.nanmean(hduRef[0].data,axis=0)
            if(type(refCombine)==type(1)):
                hduRef[0].data = hduRef[0].data[refCombine]
            
            hdus = [hduRef[0]] # Put the reference image first

        else:
            ## If the user has provided a reference image, we will use that as the reference image for alignment.
            hdus = [hduRef]
 
            
        for idx,f in enumerate(self.filenames): 
            processedHDU = preprocess(f, slicer=self.slicer)
            if(len(self.offsets_arcsecond)>0):
                ## Apply user supplied offsets.
                processedHDU.header['CRVAL1'] = processedHDU.header['CRVAL1'] + self.offsets_arcsecond[idx][0]/3600.0
                processedHDU.header['CRVAL2'] = processedHDU.header['CRVAL2'] - self.offsets_arcsecond[idx][1]/3600.0

                hdus.append(processedHDU)


            else:
                hdus.append(processedHDU)
            
        shifted_frames, shifts, mosaic_wcs, mosaic_shape = reproject_and_mosaic(
            hdus, 
            resolution=self.resolution, 
            autocorrelate = self.autocorrelate,
            autocorrelate_maskfile = self.autocorrelate_maskfile,
            correlate_mode= self.correlate_mode,
            search_size   = self.search_size,
            offset = self.offset

        )
        
        hdu         = fits.PrimaryHDU()
        hdu.data    = shifted_frames
        hdu.header  = mosaic_wcs.to_fits()[0].header
        hdu.writeto(f"{self.objid}_shifted.fits", overwrite=True)

        f = open(f"{self.objid}_{self.slicer}_shifts.list", "w" )
        for i, shift in enumerate(shifts):
            f.write(f"{shift}\n")
        f.close()
        print("Saved shifts to file:", f"{self.objid}_{self.slicer}_shifts.list")
        
        try:
            import os
            os.system(f"ds9 {self.objid}_shifted.fits")
        except Exception as e:
            print(f"Error opening ds9: {e}")
        
        self.shifts       = shifts[1:] ## DOn't need the shifts for the reference DECaLS/Panstarrs image in the array.
        self.mosaic_wcs   = mosaic_wcs
        self.mosaic_shape = mosaic_shape
        

    def removeSkyGradient(self, hdus, med_pixs = 24):
        if(len(hdus)!=len(self.skymaskFilenames)):
            raise ValueError("Number of skymasks does not match the number of filenames")
        
        for hduNum in range(len(hdus)):    
            data_foo = hdus[hduNum].data
            skyFit   = np.zeros(data_foo.shape)
            mask     = fits.getdata(self.skymaskFilenames[hduNum])
            
            print(f"Removing sky gradients for datacube-{hduNum+1}")
            
            for i in np.arange(med_pixs, data_foo.shape[0] -med_pixs, 1): ## Running sky gradient subtraction
                skyFit[i,:,:] = getSkyModel(
                    np.nanmedian(data_foo[i-med_pixs:i+med_pixs+1,:,:], axis=0),
                    mask
                )
            hdus[hduNum].data = data_foo - skyFit         
            
            ## Store the fits -- Do not use this sky model for anything important. This is only for inspection
            hduSky                      = fits.PrimaryHDU()
            hduSky.data                 = skyFit
            hduSky.header["COMMENT"]    = "Removed sky gradient in each datacube using 2D first-order polynomial"
            hduSky.writeto(self.skymaskFilenames[hduNum].replace(".fits", "_skyfits.fits"), overwrite=True)
            
            ## Save the fits for central wavelength regions to show to the user how the fits are doing
            skyFit[i,:,:] = getSkyModel(
                np.nanmedian(data_foo[data_foo.shape[0]//2-med_pixs:data_foo.shape[0]//2+med_pixs+1,:,:], axis=0),
                mask,
                plotting=True
            )
            plt.savefig(self.skymaskFilenames[hduNum].replace(".fits", "_skyfits_centralwavelength.png"))
            plt.close()
        
        return(hdus)

    def step2(self):
        hdus = []
        for idx,filename in enumerate(self.filenames): 
            processedHDU = preprocess(filename, slicer=self.slicer, cube=True) ## FLAM16 units -- default KCWI units
            # But reproject requires data to be in surface brightness units
            dx    = np.sqrt(processedHDU.header['CD1_1']**2+processedHDU.header['CD2_1']**2)*3600.
            dy    = np.sqrt(processedHDU.header['CD1_2']**2+processedHDU.header['CD2_2']**2)*3600.
            area  = dx*dy
            processedHDU.data = processedHDU.data/area

            if(len(self.offsets_arcsecond)>0):
                ## Apply user supplied offsets first
                processedHDU.header['CRVAL1'] = processedHDU.header['CRVAL1'] + self.offsets_arcsecond[idx][0]/3600.0
                processedHDU.header['CRVAL2'] = processedHDU.header['CRVAL2'] - self.offsets_arcsecond[idx][1]/3600.0

                hdus.append(processedHDU) ## FLAM16/arcsec^2 units
            else:
                hdus.append(processedHDU) ## FLAM16/arcsec^2 units
            
        ## If user has provided sky frames to use, we will use them to remove any sky gradients introduced from the IDL reduction
        if(self.skymaskFilenames is not None):
            hdus = self.removeSkyGradient(hdus)

        mosaic_data, shiftedframe_hdu = reproject_and_mosaic_cube(
            hdus          = hdus,
            shifts        = self.shifts,
            mosaic_wcs    = self.mosaic_wcs,
            mosaic_shape  = self.mosaic_shape,
            objid         = self.objid, 
        )
        
        shiftedframe_hdu.writeto(f"{self.objid}_shifted_step2.fits", overwrite=True)

        plt.figure(dpi=200)
        ScaleImage(np.nanmedian(mosaic_data, axis=0)).plot()
        plt.tight_layout()
        plt.savefig(self.objid+"_mosaic_white_light.png")
        plt.show()

        # Save to datacube
        hdu = fits.PrimaryHDU()
        hdu.data = mosaic_data
        hdu.header = self.mosaic_wcs.to_fits()[0].header
        hdrFoo = fits.getheader(self.filenames[0])
        ## Save the hdrReference as txt file for the user to inspect and use for future reference if needed.
        with open(f"{self.objid}_{self.slicer}_hdrReference.txt", "w") as f:
            for key in hdrFoo.keys():
                f.write(f"{key}: {hdrFoo[key]} -- {hdrFoo.comments[key]}\n")
    
        hdrFooComments = hdrFoo.comments

        hdu.header['WCSAXES']   = 3 ## Note that tha mosaic_wcs has only 2 dimensions.
        # hdu.header["BUNIT"]     = (hdrFoo['BUNIT'],hdrFooComments['BUNIT'])
        hdu.header["BUNIT"]     = "FLAM16/arcsec^2"
        hdu.header['CRVAL3']    = (hdrFoo['CRVAL3'],hdrFooComments['CRVAL3'])
        hdu.header['CRPIX3']    = (hdrFoo['CRPIX3'],hdrFooComments['CRPIX3'])
        hdu.header['CDELT3']    = (hdrFoo['CD3_3'],hdrFooComments['CD3_3'])
        hdu.header['CUNIT3']    = (hdrFoo['CUNIT3'],hdrFooComments['CUNIT3'])
        hdu.header['CTYPE3']    = (hdrFoo['CTYPE3'],hdrFooComments['CTYPE3'])
                
        hdu.header.remove('LONPOLE')
        hdu.header.remove('LATPOLE')
        
        hdu.header["FILES"]   = (f"{','.join(self.filenames)}", "List of files used to make the mosaic datacube")
        hdu.header["SHIFT"]   = (f"{self.objid}_{self.slicer}_shifts.list", "File containing the shifts applied to each datacube for alignment")
        hdu.header["VERSION"]   = ( "ISMgas-v1.0.5", "Version of ISMgas used to create the datacube")

        if(self.skymaskFilenames is not None):
            hdu.header["COMMENT"]   = "Removed sky gradient in each datacube using 2D first-order polynomial"
            hdu.header["MASKS"]   = (f"{','.join(self.skymaskFilenames)}", "List of sky masks used to remove sky gradients")
            
        hdu.writeto(f"{self.objid}_{self.slicer}_combined.fits", overwrite=True)
        print(f"Datacube saved as {self.objid}_{self.slicer}_combined.fits") 
