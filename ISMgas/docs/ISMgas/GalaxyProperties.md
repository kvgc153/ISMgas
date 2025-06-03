Module ISMgas.GalaxyProperties
==============================

Classes
-------

`GalaxyProperties(**kwargs)`
:   

    ### Descendants

    * ISMgas.kcwi.kcwiFunctions.kcwiAnalysis
    * ISMgas.spectra.AnalyzeSpectra.AnalyzeSpectra

    ### Methods

    `SDSS_spectra(self, index=0, search_min=0.0005, search_max=0.0005, ylim=[-1, 5])`
    :   Adapted from this example from the datalab: https://github.com/astro-datalab/notebooks-latest/blob/master/04_HowTos/SPARCL/How_to_use_SPARCL.ipynb

    `decalsFitsAndPng(self, grab=False, **kwargs)`
    :   kwargs:
        
        'grab': Grab the fits file from server or use the one that exists. False by default
        'ra': RA
        'dec': DEC
        'pixscale': default is 0.257
        'layer': default is ls-dr9 (could be hsc-dr2)
        'size': Size of output image
        
        humvi params:
        
        rscale, gscale, bscale    = kwargs.get('scale', [1,1.5,3])
        Q,alpha                   = kwargs.get('q,alpha', [1,15])
        masklevel,offset          = kwargs.get('mask,offset', [-1.0,0.0])
        saturation,backsub,vb     = kwargs.get('saturation,backsub,vb',['white',False,False])

    `fitscgi_HST(self, **kwargs)`
    :

    `plot_SDSS_spec(self, index, results, ylim=[-1, 5])`
    :   Pass an index value and the output from using client.retrieve()
        to plot the spectrum at the specified index.