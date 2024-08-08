#!/usr/bin/python
"""
This module is useful to define ColourMaps in matplotlib
Here we define the SAURON color map, which is useful
as a rainbow colour map without too much green
(good for velocity fields)
"""

"""
Importing the required modules
"""

# import sys, os, tempfile  # Standard Libraries
import numpy as np       # numpy to be used as num

try :
    import matplotlib
except ImportError:
    print('There is a problem with importing matplotlib (pylab) at initialisation')

import matplotlib.pyplot as plt
from matplotlib import colors

__version__ = '1.0.0 (July 2, 2012)'

################################################################################################
# FIXED data
################################################################################################
## Values for the interpolated Sauron colour map
xSauronCOL = np.array([1.0, 43.5, 86.0, 86.0+20, 128.5-10, 128.5, 128.5+10, 171.0-20, 171.0, 213.5, 256.0])
SauronRed =   [0.0, 0.0, 0.4,  0.5, 0.3, 0.0, 0.7, 1.0, 1.0,  1.0, 0.9]
SauronGreen = [0.0, 0.0, 0.85, 1.0, 1.0, 0.9, 1.0, 1.0, 0.85, 0.0, 0.9]
SauronBlue =  [0.0, 1.0, 1.0,  1.0, 0.7, 0.0, 0.0, 0.0, 0.0,  0.0, 0.9]

################################################################################################
# Define the Sauron colour map
################################################################################################
def get_SauronCmap() :
    """
    Define the Sauron color maps and return it 
    (as a cmap from matplotlib)
    """
    Nseg = len(xSauronCOL)
    Nm1 = np.max(xSauronCOL) - 1.0
    xSauronN = (xSauronCOL - 1.0) / Nm1
    redtup = ()
    greentup = ()
    bluetup = ()
    for i in range(Nseg) :
        redtup = redtup + ((xSauronN[i],SauronRed[i],SauronRed[i]),)
        greentup = greentup + ((xSauronN[i],SauronGreen[i],SauronGreen[i]),)
        bluetup = bluetup + ((xSauronN[i],SauronBlue[i],SauronBlue[i]),)

    cdict =  {'red': redtup, 'green': greentup, 'blue': bluetup}
    sauron_cmap = colors.LinearSegmentedColormap('Sauron_colormap',cdict,256)
    plt.register_cmap(name="Sauron", cmap=sauron_cmap)
    return sauron_cmap

