### Usage 

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
## wavrange (in km/s) defines the region used to combine the ISM lines
## Here we combine the Si-II 1260 and Si-II 1526 lines and store them in stack-combined.pkls

obj.combineISMLines(
    Nsm             = 1,
    chosen_lines    = ['Si II 1260', 'Si II 1526'],
    wavrange        = [
        [-2000,2000],
        [-2000,2000]
        
    ],
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
