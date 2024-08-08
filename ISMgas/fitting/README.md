### Usage 

```python
from ISMgas.fitting.DoubleGaussian import DoubleGaussian
dg = DoubleGaussian(
    x    = xdata,
    y    = ydata, 
    yerr = ydataerr 
)

## Set some priors for the fits 
dg.priors = {}
dg.priors["1"] ={
        "Amin"        : 0.1,
        "Amax"        : 1.0,
        "A1min"       : 0.1,
        "A1max"       : 1.0,
        "Amin_SG"     : 0.1,
        "Amax_SG"     : 1.0,
        "vmin"        : -700,
        "vmax"        : 500,
        "v1min"       : -700,
        "v1max"       : 500,
        "vmin_SG"     : -700,
        "vmax_SG"     : 500,
        "sigmin"      : 50,
        "sigmax"      : 700,
        "sig1min"     : 50,
        "sig1max"     : 700,
        "sigmin_SG"   : 50,
        "sigmax_SG"   : 700,
        "v-v1_min"    : 0, 
        "v-v1_max"    : -800,
        "cont_lvl"    : 1 ### Choose continuum level
}

## Fit a double gaussian 
results_dg = dg.fitting(double_gaussian = True)

## Fit a single gaussian 
results_sg = dg.fitting(double_gaussian = False)
```

Checkout some of the examples posted [here](examples/).