## Usage 

### Preprocessing

Save the spectra into a common format that the module can parse. 
This is essential as data is obtained from instruments on multiple telescopes and uses pipelines which output the data in multiple formats.

This module requires the following quantity and units:

| Quantity | Units |
| -------- | ------|
| Wavelength | Angstrom |
| Flux (Normalized) | Arbitrary |
| Flux(sigma) | 1 sigma uncertainity|

Use the save_spectra function to save these into a .fits file 

```python
from ISMgas.SupportingFunctions import save_spectra
save_spectra(
    wave, 
    flux, 
    error, 
    fileName = 'stack', 
    folderPrefix = ''
)

## Spectra saved as stack.fits
```

### Analysis

```python
from ISMgas.spectra.AnalyzeSpectra import AnalyzeSpectra
from ISMgas.SupportingFunctions import load_pickle, plotWithError

obj = AnalyzeSpectra(
    objid           = "stack",
    spec_filename   = "stack.fits",
    R               = 4000,
    zs              = 0
)

## plot the spectra -- stores the plot in localdrive as stack-smooth-spectra.png
## Here Nsm controls the smoothing

obj.plotSmoothedSpectra(
    Nsm  = 3,
    xlim = [1200,1600],
    ylim = [0,1.5],    
)

## Choose some ISM lines and combine them using ivar weighting
## wavrange (in km/s) defines the region used to combine the ISM lines via interpolation.
## Here we combine the Si-II 1260 and Si-II 1526 lines and store them in stack-combined.pkls

obj.combineISMLines(
    Nsm             = 1,
    chosen_lines    = ['Si II 1260', 'Si II 1526'],
    wavrange        = [
        [-2000,2000],
        [-2000,2000]
        
    ],

## The code combines the lines through interpolation. 
## Specify the range using the start_interp, end_interp and delta_interp (sampling). 
## start_interp    = -2000, end_interp      = 2000 ,delta_interp    = 25 -- Sample between -2000, 2000 at 25 km/s interval.
    start_interp    = -2000,
    end_interp      = 2000,
    delta_interp    = 25
)

## Plot the combined lines

foo = load_pickle('stack-combined.pkls')
plotWithError(
    foo['source_wav'],
    foo['source_flux'],
    foo['source_error']
)

plt.ylim([0,1.3])
```
