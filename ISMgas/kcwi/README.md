## Combining KCWI data (taken on the same night)

This module is used to combine observations of a target taken on a given night. 

Please ensure that the reduced datacubes have units of `FLAM16` and the wavelength axis has units of `Angstrom` before using this module, this is not enforced in code. 

The code will also fail if the datacubes have different wavelength arrays (for example, if they are taken on different nights/setups). 
While the code presented in this module will work on both the blue and red side datacubes, note that it has only been rigourously tested on the KCWI blue channel observations (red side is still being tested).

Please report any issues and bug fixes via a PR.

```python
from ISMgas.kcwi.kcwiReduction import kcwiRedux
import astropy.units as u

filenames = [
    "redux/kb250427_00082_icubes.fits",
    "redux/kb250427_00083_icubes.fits",
]

## Optional : Sky gradient removal 
## This step uses the provided masks to correct for nonzero residual background with a spatial gradient that is present in the datacubes. Use them as needed. 
skymaskFilenames = [
    "masks/kb250427_00082_icubes_mask.fits",
    "masks/kb250427_00083_icubes_mask.fits",
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
    skymaskFilenames = skymaskFilenames, ## Remove this if you don't want the sky gradient removal
    autocorrelate=True, ## Uses autocorrelation to align the datacubes to the DECaLS image. If you turn it off the WCS information in the headers are used without any correction.
    grab = True ## Dowloads the DECaLS fits image if True. Set to False if you are  re-running your code. 
)

obj.step1()
## By default, step1 uses a mean of the g,r and z filters in the DECaLS image to align the datacubes BUT if your target is only bright in the blue filter for example, you can also select the filter to use for alignment by using the refCombine keyword.
#obj.step1(refCombine=0)  # 0 -- g band, 1 -- r band, 2 -- z band.

obj.step2()
```

#### What does it do?

**Step 1** 
- Downloads a DECaLS image of the field
- Finds offset shifts from the DECaLS image to the datacubes using autocorrelation

**Step 2** 
- Reprojects all the datacubes using the `reproject` package to the chosen pixelsize.
- The final datacubes are aligned to the DECaLS image and combined using a mean. The final datacube has the filename - `{objid}_{slicer}_combined.fits`