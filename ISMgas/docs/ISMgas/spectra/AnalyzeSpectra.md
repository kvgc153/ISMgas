Module ISMgas.spectra.AnalyzeSpectra
====================================
Class handling i/o of spectra and making a velocity profile.
This code has been rigourously tested with data obtained from ESI and MagE. 
Feel free to try it out with other instruments. If you run into any issues, 
please report it on the repo

Classes
-------

`AnalyzeSpectra(**kwargs)`
:   Class handing the analysis of 1D spectra.  
    Example usage :
    ```
    obj = AnalyzeSpectra(
        objid           = "stack",
        spec_filename   = "stack.fits",
        R               = 4000,
        zs              = 0
    )
    
    
    obj.combineISMLines(
        Nsm             = 1,
        chosen_lines    = ['Si II 1260', 'Si II 1526'],
        wavrange        = [
            [-2000,2000],
            [-2000,2000]
            
        ],
    )
    foo = load_pickle('stack-combined.pkls')
    plotWithError(
        foo['source_wav'],
        foo['source_flux'],
        foo['source_error']
    )
    
    plt.ylim([0,1.3])
    
    
    ```

    ### Ancestors (in MRO)

    * ISMgas.GalaxyProperties.GalaxyProperties

    ### Static methods

    `smooth_and_rebin(specDict, wavmin=-1401, wavmax=1401, smooth_sigma=3, rebin_shape=(28,))`
    :   Example usage: 
        ## LRIS (R=600) ##
        - First smooth by 8 pixels (200 km/s) --> FWHM ~500 km/s (R=600)
        - Then rebin the array which is currently of shape (112,)---> (14,). 
        - Therefore 14 elements are used to represent the wavelength range -1400, 1400 km/s
        
        ## R=1000 ##
        - First smooth by 5 pixels (125 km/s) which is FWHM ~300 km/s (R=1000)
        - Then rebin the array which is currently of shape (100,)---> (20). 
        - Therefore 20 elements are used to represent the wavelength range -1250, 1250 km/s
        
        
        ## KCWI (R=2000) ##
        - First smooth by 3 pixels (75 km/s) which is FWHM ~ 175 km/s (R=2000)
        - Then rebin the array which is currently of shape (112,)---> (28,). 
        - Therefore 28 elements are used to represent the wavelength range -1400, 1400 km/s

    ### Methods

    `combineISMLines(self, Nsm, chosen_lines, wavrange=[], xlim=[-2000, 2000], ylim=[0, 2], start_interp=-2000, end_interp=2000, delta_interp=25, state='lowion', normScaleFactor=1, roi=[])`
    :   returns a weighted average of all the lines chosen by the user

    `dataloader(self, Nsm, state, wavrange=[], plotting=False, chosen_lines=0, start_interp=-2000, end_interp=2000, delta_interp=25, xlim=[-2000, 2000], ylim=[0, 2], interpolate_spec=True)`
    :   Handles importing data from pre-processed fits file. 
        returns (wavelength, flux, sigma, line_name)

    `fetchData(self)`
    :

    `plotCombinedLines(self, xlim=[-2000, 2000], ylim=[0, 2])`
    :

    `plotContinuumFit(self, xlim=[-2000, 2000], ylim=[0, 2])`
    :

    `plotLines(self, Nsm, xlim=[-2000, 2000], ylim=[0, 2], state='lowion')`
    :   Plots the individual ISM lines from spectra. Plots the raw spectra without interpolation.

    `plotSmoothedSpectra(self, Nsm, display='python', xlim=[None, None], ylim=[None, None], printLinesUsed=False, plotDeflectorLines=False, plotStellar=False)`
    :   Plots weighted average spectra after smooted by Nsm sized kernel
        This can be displayed on a simple python matplotlib.
        
        
        - Nsm = number of elements to average over
        - display = 'python' or 'js' - displays the spectra using matplotlib or plotly
        - linelists_highz = ISM lines
        - linelists_stellar = Stellar lines
        - linelists_general = rough lines spanning the entire spectral range.
        
        After plotting, returns the wavlength,flux and sigma arrays used for plotting

    `plotVerticalStack(self, xlim=[-2000, 2000], ylim=[0, 2])`
    :