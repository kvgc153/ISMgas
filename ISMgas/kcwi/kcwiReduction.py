from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.signal import correlate2d
from scipy.ndimage import shift
from astropy.io import fits
from astropy.visualization import ZScaleInterval, ImageNormalize, PercentileInterval


from reproject import reproject_exact, reproject_interp
from reproject.mosaicking import find_optimal_celestial_wcs

from ISMgas.kcwi.kcwiFunctions import *
from ISMgas.visualization.fits import ScaleImage


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

def preprocess(filename, slicer = 'medium'):
    hdu1 = fits.PrimaryHDU()
    WCS1 = WCS(fits.getheader(filename))
    WCS1 = WCS1.dropaxis(2)
    hdr = WCS1.to_fits()[0].header
    
    frame =  fits.getdata(filename)
    if(slicer=='small'):
        frame[:,0:29,:] = np.nan
        frame[:,163:,:] = np.nan

        frame[:,:,:11] = np.nan
        frame[:,:,32:] = np.nan       
    
    if(slicer=='medium'):
        # Pad the edges with zeros, to remove noisy data
        frame[:,0:15,:] = np.nan
        frame[:,80:,:] = np.nan

        frame[:,:,0:5] = np.nan
        frame[:,:,-5:] = np.nan
        
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
    # WCS1 = WCS(fits.getheader(filename))
    # WCS1 = WCS1.dropaxis(2)
    # hdr = WCS1.to_fits()[0].header
    
    hdr   = fits.getheader(filename) 
    frame =  fits.getdata(filename)
    
    if(slicer=='small'):
        frame[:,0:29,:] = np.nan
        frame[:,163:,:] = np.nan

        frame[:,:,:11] = np.nan
        frame[:,:,32:] = np.nan        
        
    if(slicer=='medium'):
        # Pad the edges with zeros, to remove noisy data

        frame[:,0:14,:] = np.nan
        frame[:,80:,:] = np.nan

        frame[:,:,0:5] = np.nan
        frame[:,:,-5:] = np.nan
        
    elif(slicer=='large'):
        # Pad the edges with zeros, to remove noisy data
        frame[:,0:14,:] = np.nan
        frame[:,79:,:] = np.nan

        frame[:,:,:2] = np.nan
        frame[:,:,25:] = np.nan
    
    hdu1.data   = frame
    hdu1.header = hdr

    return(hdu1)



def reproject_and_mosaic(hdus, method='exact', apply_shift=True, resolution=None, return_shifted_frames=False):
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
    if(resolution is None):
        mosaic_wcs, mosaic_shape = find_optimal_celestial_wcs(hdus)

    elif(resolution is not None):
        mosaic_wcs, mosaic_shape = find_optimal_celestial_wcs(hdus, resolution=resolution)
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
    
    shifted_frames  = []
    shifted_frames.append(ref_data)  # Store the reference frame

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
        
        shifted_frames.append(shifted_data)  # Store the shifted frame

    # Normalize
    with np.errstate(divide='ignore', invalid='ignore'):
        mosaic_data = np.where(weight_map > 0, combined_data / weight_map, np.nan)

    if(return_shifted_frames):
        return shifted_frames, mosaic_wcs
    else:
        return mosaic_data, mosaic_wcs


def reproject_and_mosaic_cube(hdus, resolution, spectral_axis=0, parallel=1, method='exact', apply_shift=True, check_samewave=True):
    """
    Reproject multiple data cubes onto a common WCS frame that covers all of them,
    align them spatially using cross-correlation (once), and combine them into a single cube mosaic.

    Parameters
    ----------
    hdus : list of astropy.io.fits.PrimaryHDU or ImageHDU
        List of 3D data cube HDUs with WCS.
    resolution : astropy Quantity
        Dictates what resolution the frames must be projected to.
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
    
    
    ## Check if all cubes have same wavelength axis
    print("Checking if all cubes have the same wavelength axis...")
    if check_samewave:
        for i in range(1, len(hdus)):
            if not kcwi_check_samewave(hdus[0].header, hdus[i].header):
                raise ValueError(f"The wavelength axes of the {0} and {i} cubes are not the same. Fix this before proceeding.")



    # Use median images for spatial alignment
    spatial_hdus = []
    for hdu in hdus:
        median_image = np.nanmedian(hdu.data, axis=spectral_axis)
        wcsDrop = WCS(hdu.header).dropaxis(2)  # Drop the spectral axis for spatial WCS
        hdu.header = wcsDrop.to_fits()[0].header       
        spatial_hdus.append(fits.ImageHDU(median_image, header=hdu.header))

    mosaic_wcs, mosaic_shape = find_optimal_celestial_wcs(spatial_hdus, resolution=resolution)
    
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
            if(i%200==0):
                print(f"Reprojecting slice {i + 1}/{n_spectral} of datacube-{ndatacube+1}...")
            if spectral_axis == 0:
                slice_data = cube_data[i, :, :]
            else:
                ## placeholder for more general implmentation.
                raise ValueError("Unsupported spectral_axis value. Must be 0.")

            slice_hdu = fits.ImageHDU(slice_data, header=hdu.header)

            if method == 'exact':
                reproj_slice, footprint = reproject_exact(slice_hdu, mosaic_wcs, shape_out=mosaic_shape, parallel=parallel)
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
    # Use np.errstate to suppress warnings for division by zero or invalid operations
    with np.errstate(divide='ignore', invalid='ignore'):
        # Create an empty array of the same shape as mosaic_cube to hold the result
        result = np.empty_like(mosaic_cube)

        # For all elements where weight_cube > 0, divide mosaic_cube by weight_cube
        result[weight_cube > 0] = mosaic_cube[weight_cube > 0] / weight_cube[weight_cube > 0]

        # For all elements where weight_cube <= 0, set result to NaN
        result[weight_cube <= 0] = np.nan

        # Assign the final result back to mosaic_cube
        mosaic_cube = result        

    return mosaic_cube, mosaic_wcs
