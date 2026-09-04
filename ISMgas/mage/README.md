
## Steps for reduction 

```python
folder = '../agel142719_pa1/'
objid  = "agel142719_pa1"

obj = mageRedux(
    objid = objid,
    folder = folder,
    extractWindow = {'min': 4, 'max':6},
    stdstar='ltt3864'
)

obj.extractAndPlot() ## Extracts spectra from each echelle order

obj.correctSensitivity() ## Flux calibration 

combined =  obj.combine_echelle_orders() ## OPTIONAL:  Combines the echelles orders into a 1D spectra - great for quick look + basic analysis

```
