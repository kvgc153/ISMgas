Module ISMgas.kcwi.kcwiReduction
================================

Functions
---------

`kcwi_check_samewave(hdr0, hdr1)`
:   This code is adapted from kcwikit -- https://github.com/yuguangchen1/KcwiKit/blob/master/kcwikit/kcwi/kcwi.py
    Check if the wavelength axes are the same in two headers.
    
    Args:
        hdr0 (astropy.io.fits.header) - input header #0
        hdr1 (astropy.io.fits.header) - input header #1
    
    Returns:
        boolean: whether wave axes are the same

`padAndAlign(cubes, newOutputShape, centroids=[], idx=[500, -500], method='mean')`
:   

`preprocess(filename, slicer='medium')`
:   

`preprocessCube(filename, slicer='medium')`
:   

`reproject_and_mosaic(hdus, method='exact', apply_shift=True, resolution=None)`
:   Reproject multiple 2D images onto a common WCS frame, 
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

`reproject_and_mosaic_cube(hdus, spectral_axis=0, parallel=1, method='exact', apply_shift=True, check_samewave=True)`
:   Reproject multiple data cubes onto a common WCS frame that covers all of them,
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