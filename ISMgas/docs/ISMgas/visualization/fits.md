Module ISMgas.visualization.fits
================================

Functions
---------

`mockData2D()`
:   Generates a 2D mock image.

`regionToMask(filename, data, kwargs={'format': 'ds9'})`
:   Generates a mask from a region file.
    
    Args:
        filename (str): Region file name.
        data (np.array): Data array.
        kwargs (dict, optional): _description_. Defaults to {'format':'ds9'}.
    
    Returns:
        np.array: Mask array.

Classes
-------

`ScaleImage(image: numpy.array, scale: str = 'percentile', scale_kwargs: dict = {'percentile': 95}, cmap: str = 'gray')`
:   This class is used to scale and plot astronomical images.
    
    Args:
        image (numpy.array): _description_
        scale (str, optional): _description_. Defaults to 'percentile'.
        scale_kwargs (_type_, optional): _description_. Defaults to {'percentile':95}.
        cmap (str, optional): _description_. Defaults to 'gray'.

    ### Methods

    `plot(self, origin='lower')`
    :