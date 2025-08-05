## Combining KCWI data (taken on the same night)

This module requires the reduced datacubes to have units of FLAM16 and the wavelength axis to have units of Angstrom. 
Ensure that this is the case before running the code, the module will not enforce this. 

The code will also fail if the datacubes have different wavelength arrays (for example, if they are taken on different nights/setups). 
While the code presented in this module will work on both the blue and red side datacubes, note that it has only been rigourously tested on the KCWI blue channel observations. 

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
- Reprojects all the datacubes using the `reproject` package to the chosen pixelsize.
- The final datacubes are aligned to the DECaLS image and combined using a mean. The final datacube has the filename - `{objid}_{slicer}_combined.fits`