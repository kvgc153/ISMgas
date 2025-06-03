Module ISMgas.kcwi.kcwiFunctions
================================

Classes
-------

`kcwiAnalysis(**kwargs)`
:   

    ### Ancestors (in MRO)

    * ISMgas.GalaxyProperties.GalaxyProperties

    ### Methods

    `humviPNG(self, bwave=[500, 1000], gwave=[800, 1200], rwave=[1200, 1500])`
    :   This function will create a png file using the humvi package.
        rwave, gwave, bwave are the wavelength ranges for the red, green, and blue channels.

    `overlayMask(self, **kwargs)`
    :

    `return_meancube(self)`
    :

    `return_mediancube(self)`
    :

    `spectraUnderMask(self, method='sum')`
    :   Computes the spectra under the mask. The method can be either sum or mean.
        
        Args:
            method (str, optional): Method to collapse mask spectra. Defaults to 'sum'.

    `whiteLightImage(self, **kwargs)`
    :