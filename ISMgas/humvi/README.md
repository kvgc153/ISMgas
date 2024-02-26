
Source code is from : https://github.com/drphilmarshall/HumVI but it runs only with python2. For this repo, I modified it to run in python3.

### Usage 

humvi.compose(rfile, gfile, bfile, scales=(rscale,gscale,bscale), Q=Q, alpha=alpha, masklevel=masklevel, saturation=saturation, offset=offset, backsub=backsub, vb=vb, outfile=outfile)

```
humvi.compose("humvi_R.fits", "humvi_G.fits", "humvi_B.fits", 
                  scales=(1,1.4,2), 
                  Q=3, 
                  alpha=0.4, 
                  masklevel=-1, 
                  saturation='white', 
                  offset=0, 
                  backsub=False, 
                  vb=False, 
                  outfile=outfile)
                  
```                  

### Changes
* Sept-02: Changed code to work for python3
