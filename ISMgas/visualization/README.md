
This module contains useful tools for visualizing astronomical data.

### Scaling astronomy images

```python
from ISMgas.visualization.fits import ScaleImage

s = ScaleImage(
    image = img, ## This is a 2D array
    scale= 'percentile', 
    scale_kwargs = {'percentile':99.99}, 
    cmap = 'viridis'
)

s.plot()
```
Currently supports only 'percentile' but more options will be added shortly.

Example notebook -- [Link](examples/example-visualization.ipynb)

### Converting ds9 regions to mask

In order to do this right, you need to supply the maskfilename and data that the mask belongs to. In the example below, maskFilename is the mask file corresponding to 'data'.

```python
from ISMgas.visualization.fits import regionToMask

mask2D = regionToMask(
    maskFilename, 
    data, 
    kwargs={'format':'ds9'}
)
```