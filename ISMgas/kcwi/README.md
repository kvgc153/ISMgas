## Reducing KCWI data 

```python
from ISMgas.kcwi.kcwiReduction import kcwiRedux
import astropy.units as u

filenames = [
    "redux/kb250427_00082_icubes.fits",
    "redux/kb250427_00083_icubes.fits",
]

# Required parameters #
objid = "DESI231"
ra  = 231.2874	
dec = 42.4646
resolution = 0.3*u.arcsecond ## Pixelscale (arcsecond/pixel)
size= 100 ## Size of the field  (in pixels)
slicer='medium' ## small, medium, large

obj  = kcwiRedux(
    objid, 
    ra, dec, 
    resolution, size, 
    filenames, 
    slicer, 
    autocorrelate=True,
    grab = True ## Dowloads the DECaLS fits image if True. Set to False if you are  re-running your code. 
)

obj.step1()
obj.step2()
```

#### What does it do?

**Step 1** 
- Downloads a DECaLS image of the field
- Finds offset shifts from the DECaLS image to the datacubes using autocorrelation

**Step 2** 
- Reprojects all the datacubes using the `reproject` package.
- The final datacubes are aligned to the DECaLS image and.