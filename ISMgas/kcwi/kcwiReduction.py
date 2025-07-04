from scipy.signal import correlate2d
from scipy.ndimage import shift
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt

from reproject import reproject_exact, reproject_interp
from reproject.mosaicking import find_optimal_celestial_wcs

from symfit.core.minimizers import DifferentialEvolution,BFGS,BasinHopping
from symfit import Poly, variables, parameters, Model, Fit, cos,GreaterThan,LessThan

from ISMgas.visualization.fits import ScaleImage
from ISMgas.GalaxyProperties import GalaxyProperties
from ISMgas.kcwi.kcwiFunctions import kcwiAnalysis


def preprocess(filename, slicer = 'medium', cube = False):
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

def reproject_and_mosaic(hdus, method='exact', autocorrelate=False, autocorrelate_maskfile= None, correlate_mode='full',  resolution=None):
    """
    Reproject multiple 2D images onto a common WCS frame, 
    align them using 2D cross-correlation, and combine into a mosaic.
    """
    if len(hdus) == 0:
        raise ValueError("No HDUs provided.")
    if method not in ['exact', 'interp']:
        raise ValueError("method must be 'exact' or 'interp'.")
    ## If a autocorrelate mask is provided,  make userMask to apply later
    if(autocorrelate_maskfile is not None):
        userMask = fits.getdata(autocorrelate_maskfile )
        userMask = userMask.astype(float)
        userMask[userMask==0] = np.nan

    # Choose the reprojection function
    reproj_func = reproject_exact if method == 'exact' else reproject_interp

    # Find optimal WCS
    if(resolution is None):
        mosaic_wcs, mosaic_shape = find_optimal_celestial_wcs(hdus)

    elif(resolution is not None):
        mosaic_wcs, mosaic_shape = find_optimal_celestial_wcs(hdus, resolution=resolution)
        
    # Choose the first image as reference -- this is the DECaLS/SDSS/Panstaars image.
    ref_hdu     = hdus[0]
    ref_data, _ = reproj_func(ref_hdu, mosaic_wcs, shape_out=mosaic_shape)
    
    shifted_frames  = []
    shifts = [(0.0, 0.0)]  # Store shifts for each frame
    
    shifted_frames.append(ref_data)  # Store the reference frame

    for hdu in hdus[1:]:
        # Reproject the target image
        target_data, _ = reproj_func(hdu, mosaic_wcs, shape_out=mosaic_shape)

        # Cross-correlation to find shift
        if(autocorrelate_maskfile is not None):
            valid_mask = (~np.isnan(ref_data)) & (~np.isnan(target_data)) & (~np.isnan(userMask))
        else:
            valid_mask = (~np.isnan(ref_data)) & (~np.isnan(target_data))

        if np.sum(valid_mask) == 0:
            continue  # Skip if no overlap
        
        if(autocorrelate):
            corr = correlate2d(
                np.nan_to_num(ref_data) * valid_mask, 
                np.nan_to_num(target_data) * valid_mask,
                mode=correlate_mode
            )

            shift_y, shift_x = np.array(np.unravel_index(np.argmax(corr), corr.shape)) - np.array(corr.shape) // 2

            # Apply the shift
            shifted_data = shift(target_data, shift=(shift_y, shift_x), order=1, mode='constant', cval=np.nan)
            shifts.append((shift_y, shift_x))  # Store the shift
            
            
        else:
            shifted_data = target_data
            shifts.append((0.0, 0.0))  # No shift applied
            
       
        shifted_frames.append(shifted_data)  # Store the shifted frame

    return shifted_frames, shifts, mosaic_wcs, mosaic_shape

def reproject_and_mosaic_cube(hdus, mosaic_wcs, mosaic_shape, spectral_axis=0, method='exact', shifts=[]):
    """
    Reproject multiple data cubes onto a common WCS frame that covers all of them,
    align them spatially using cross-correlation (once), and combine them into a single cube mosaic.
    """
    ## Begin Checks ## 
    if len(hdus) == 0:
        raise ValueError("No HDUs provided.")

    if method not in ['exact', 'interp']:
        raise ValueError("method must be 'exact' or 'interp'.")

    ## Check if all cubes have same wavelength axis
    for i in range(1, len(hdus)):
        if not kcwi_check_samewave(hdus[0].header, hdus[i].header):
            raise ValueError(f"The wavelength axes of the {0} and {i} cubes are not the same. Fix this before proceeding.")

    ## Check if len(shifts) == len(hdus
    if len(shifts) != len(hdus):
        raise ValueError("The length of shifts must match the number of HDUs provided.")
    
    ## End checks ## 

    # Initialize output cubes
    n_spectral  = hdus[0].data.shape[spectral_axis]
    mosaic_cube = np.full((n_spectral, *mosaic_shape), np.nan)
    weight_cube = np.zeros((n_spectral, *mosaic_shape), dtype=float)

    # Now apply shifts and reproject full cubes
    for ndatacube, (hdu, (shift_y, shift_x)) in enumerate(zip(hdus, shifts)):
        cube_data = hdu.data
        
        wcsDrop = WCS(hdu.header).dropaxis(2)  # Drop the spectral axis from WCS
        hdu.header = wcsDrop.to_fits()[0].header  

        for i in range(n_spectral):
            if(i%200==0):
                print(f"Reprojecting slice {i + 1}/{n_spectral} of datacube-{ndatacube+1}...")
            if spectral_axis == 0:
                slice_data = cube_data[i, :, :]
            else:
                raise ValueError("Unsupported spectral_axis value. Must be 0.")

 
            slice_hdu = fits.ImageHDU(slice_data, header=hdu.header)

            if method == 'exact':
                reproj_slice, footprint = reproject_exact(slice_hdu, mosaic_wcs, shape_out=mosaic_shape)
            else:
                reproj_slice, footprint = reproject_interp(slice_hdu, mosaic_wcs, shape_out=mosaic_shape)
                

            # Shift the reprojected slice spatially based on the calculated shifts (shift_y, shift_x)
            # The shift is applied using interpolation (order=1), and out-of-bounds areas are filled with 0
            shifted_slice = shift(reproj_slice, shift=(shift_y, shift_x), order=1, mode='constant', cval=0)

            # Similarly, shift the footprint (validity mask) of the reprojected slice
            # This ensures that the validity of the shifted slice is correctly aligned
            shifted_footprint = shift(footprint, shift=(shift_y, shift_x), order=1, mode='constant', cval=0.0)

            # Identify valid pixels in the shifted footprint (non-zero values indicate valid contributions)
            valid = shifted_footprint > 0

            # Add the shifted slice to the mosaic cube at valid pixel locations
            # Replace NaN values with zeros to ensure proper addition
            mosaic_cube[i][valid] = np.nan_to_num(mosaic_cube[i][valid]) + np.nan_to_num(shifted_slice[valid])

            # Increment the weight cube to track the number of contributions for each pixel
            weight_cube[i][valid] += 1

    # Normalize the mosaic cube by dividing by the weight cube
    # This ensures that the final mosaic is an average of all contributing slices
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.empty_like(mosaic_cube)
        
        # For all elements where weight_cube > 0, divide mosaic_cube by weight_cube
        result[weight_cube > 0] = mosaic_cube[weight_cube > 0] / weight_cube[weight_cube > 0]

        # For all elements where weight_cube <= 0, set result to NaN
        result[weight_cube <= 0] = np.nan

        mosaic_cube = result        

    return mosaic_cube

def getSkyModel(flux, mask, plotting=False, verbose=False):
    x, y, z                = variables('x, y, z')
    c0,c1, c2,c3,c4,c5,c6 = parameters('c0,c1,c2,c3,c4,c5,c6')
    # Make a polynomial. Note the `as_expr` to make it symfit friendly.
    model_dict = {
        z: Poly( {(0, 0): c0,(1, 0): c1, (0, 1): c2,(1, 1): c3}, x ,y).as_expr()
    }
    model = Model(model_dict)


    obj_flux = mask*flux

    xdata,ydata = np.where(~np.isnan(obj_flux))  ## Ensures that the masked out region is not used for fitting
    zdata = obj_flux[xdata,ydata]
    mask_nonzero =  flux!=0

    # Perform the fit
    fit = Fit(model, x=xdata, y=ydata, z=zdata)
    fit_result = fit.execute()

    zfit = model(x=xdata, y=ydata, **fit_result.params)
    if verbose:
        print(fit_result)

    sky_model = np.zeros(obj_flux.shape)
    sky_model[xdata,ydata] = zfit
    sky_model = sky_model* mask_nonzero
    
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


        #################################################
        plt.subplot(1,3,3)
        x,y = np.where(flux*mask!=0)
        z = (flux*mask)[x,y]
        plt.hist(z,label='Before sky correction',alpha=0.6)

        z = ((flux-sky_model)*mask)[x,y]
        plt.hist(z,label='After sky correction',alpha = 0.6)
        plt.legend()
        
        plt.tight_layout()


    xdata,ydata = np.where(~np.isnan(obj_flux))  
    zfit = model(x=xdata, y=ydata, **fit_result.params)
    sky_model = np.zeros(obj_flux.shape)
    sky_model[xdata,ydata] = zfit
    sky_model = sky_model* mask_nonzero        
    
    return(sky_model)

class kcwiRedux:
    def __init__(self, 
                 objid, ra, dec, 
                 resolution, size, 
                 filenames, slicer, 
                 grab=False,
                 autocorrelate=True, autocorrelate_maskfile = None, correlate_mode='full', 
                 skymaskFilenames = None
        ):
        self.filenames = filenames
        self.skymaskFilenames = skymaskFilenames
        
        self.objid = objid
        self.slicer = slicer
        self.resolution = resolution
        
        gg = GalaxyProperties(
            ra= ra,
            dec = dec,
            objid = objid + "_DECALS"
        )
        gg.decalsFitsAndPng(grab=grab, pixscale = resolution.value, size=size)

        ## Autocorrelate and auto align all the datcubes -- great for quick redux.
        self.autocorrelate = autocorrelate
        self.autocorrelate_maskfile = autocorrelate_maskfile
        self.correlate_mode = correlate_mode
        
        
        ## Show user color image of all the datacubes that are going to be reduced 
        self.showWhiteLightImage()
        
    def showWhiteLightImage(self):
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
        

    def step1(self, refCombine='mean'):
        hduRef = fits.open( self.objid + "_DECALS.fits")
        
        hduRef[0].header = WCS(hduRef[0].header).dropaxis(2).to_fits()[0].header
        
        ## By default we will use the mean of the DECaLS image 
        if(refCombine=='mean'):
            hduRef[0].data = np.nanmean(hduRef[0].data,axis=0)
        
        ## But if for some reason that fails (too bright targets in the red). then you can pass an integer to specify the band
        if(type(refCombine)==type(1)):
            hduRef[0].data = hduRef[0].data[refCombine]
        
        hdus = [hduRef[0]]
        for f in self.filenames:
            hdus.append(preprocess(f, slicer=self.slicer))
            
        shifted_frames, shifts, mosaic_wcs, mosaic_shape = reproject_and_mosaic(
            hdus, 
            resolution=self.resolution, 
            autocorrelate = self.autocorrelate,
            autocorrelate_maskfile = self.autocorrelate_maskfile,
            correlate_mode= self.correlate_mode
        )
        hdu = fits.PrimaryHDU()
        hdu.data = shifted_frames
        hdu.header = mosaic_wcs.to_fits()[0].header
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
        
        self.shifts = shifts[1:]
        self.mosaic_wcs = mosaic_wcs
        self.mosaic_shape = mosaic_shape
        

    def removeSkyGradient(self,hdus, med_pixs = 24):
        if(len(hdus)!=len(self.skymaskFilenames)):
            raise ValueError("Number of skymasks does not match the number of filenames")
        
        for hduNum in range(len(hdus)):    
            data_foo = hdus[hduNum].data
            skyFit   = np.zeros(data_foo.shape)
            mask     = fits.getdata(self.skymaskFilenames[hduNum])
            
            print(f"Removing sky gradients for datacube-{hduNum+1}")
            
            for i in np.arange(med_pixs, data_foo.shape[0] -med_pixs, 1): ## Runing sky subtraction
                skyFit[i,:,:] = getSkyModel(
                    np.nanmedian(data_foo[i-med_pixs:i+med_pixs+1,:,:], axis=0),
                    mask
                )
            hdus[hduNum].data = data_foo - skyFit         
            
            ## Store the fits 
            hduSky = fits.PrimaryHDU()
            hduSky.data = skyFit
            hduSky.header["COMMENT"] = "Removed sky gradient in each datacube using 2D first-order polynomial"
            hduSky.writeto(self.skymaskFilenames[hduNum].replace(".fits", "_skyfits.fits"), overwrite=True)
            
            ## Save the fits for central wavelength regions
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
        for filename in self.filenames:
            hdus.append(preprocess(filename, slicer=self.slicer, cube=True))
            
        ## If user has provided sky frames to use, we will use them to remove any sky gradients introduced from the IDL reduction
        if(self.skymaskFilenames is not None):
            hdus = self.removeSkyGradient(hdus)

        mosaic_data = reproject_and_mosaic_cube(
            hdus=hdus, shifts=self.shifts, mosaic_wcs=self.mosaic_wcs, mosaic_shape=self.mosaic_shape
        )

        plt.figure(dpi=200)
        ScaleImage(np.nanmedian(mosaic_data, axis=0)).plot()
        plt.show()

        # Save to datacube
        hdu = fits.PrimaryHDU()
        hdu.data = mosaic_data
        hdu.header = self.mosaic_wcs.to_fits()[0].header
        hdrFoo = fits.getheader(self.filenames[0])
        hdrFooComments = hdrFoo.comments

        hdu.header['WCSAXES'] = 3 ## Note that tha mosaic_wcs has only 2 dimensions.
        hdu.header["BUNIT"]  = (hdrFoo['BUNIT'],hdrFooComments['BUNIT'])
        hdu.header['CRVAL3'] = (hdrFoo['CRVAL3'],hdrFooComments['CRVAL3'])
        hdu.header['CRPIX3'] = (hdrFoo['CRPIX3'],hdrFooComments['CRPIX3'])
        hdu.header['CDELT3'] = (hdrFoo['CD3_3'],hdrFooComments['CD3_3'])
        hdu.header['CUNIT3'] = (hdrFoo['CUNIT3'],hdrFooComments['CUNIT3'])
        hdu.header['CTYPE3'] = (hdrFoo['CTYPE3'],hdrFooComments['CTYPE3'])
                
        hdu.header.remove('LONPOLE')
        hdu.header.remove('LATPOLE')
        
        hdu.header["COMMENT"] = f"Files used: {','.join(self.filenames)}"
        hdu.header["COMMENT"] = f"Shifts saved to {self.objid}_{self.slicer}_shifts.list"
        hdu.header["COMMENT"] = "ISMGas version: v1.0.3"
        
        if(self.skymaskFilenames is not None):
            hdu.header["COMMENT"] = "Removed sky gradient in each datacube using 2D first-order polynomial"
            hdu.header["COMMENT"] = f"Sky mask files used: {','.join(self.skymaskFilenames)}"

        hdu.writeto(f"{self.objid}_{self.slicer}_combined.fits", overwrite=True)
        print(f"Datacube saved as {self.objid}_{self.slicer}_combined.fits") 
