Module ISMgas.nires.NIRESredux
==============================

Functions
---------

`initNIRES(filenames)`
:   Returns a dict with files to reduce.
    
    Args:
        filenames (_type_): _description_
    
    Returns:
        _type_: _description_

Classes
-------

`NIRESredux(**kwargs)`
:   

    ### Methods

    `clean(self)`
    :

    `findWavelengthOffset(self, order='sp4', delta=40, Amin=630, Amax=675, offsetWav=0, scalesky=10, figsize=(20, 7), ylim=[None, None])`
    :

    `make2DZoomIn(self)`
    :   Plots a zoom in of the 2D image for the lines

    `plotReducedSpectra(self, guessZ, detailed=False)`
    :

    `plot_emission_lines(self, z, detailed=False)`
    :

    `redux1D(self, order='sp4', delta=40, Amin=630, Amax=655, scaleLimits=[20, 85], waveLimits=[1, 99.99], z=2, scaleAlpha=1)`
    :

    `redux2D(self, mode='autox', bk='', sp='')`
    :   Run nsx in autox mode

    `save(self)`
    :

    `slitPosition2D(self)`
    :   Make a 2D image of the slit position