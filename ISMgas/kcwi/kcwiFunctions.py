from astropy.io import ascii
import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.signal import medfilt
from astropy.io import fits
from astropy.table import Table
from scipy.constants import c
c_kms = c*1e-3

import logging
from scipy import interpolate
from IPython.display import Image, display
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import display,Image
from matplotlib.patches import Rectangle
import humvi
import shutil

## Custom packages
from ISMgas.fitting.DoubleGaussian import *
from ISMgas.GalaxyProperties import *
from ISMgas.linelist import linelist_highz,linelist_SDSS
from ISMgas.SupportingFunctions import display_image


class kcwiAnalysis(GalaxyProperties):

    def __init__(self,**kwargs):

        GalaxyProperties.__init__(self,**kwargs)

        self.fileNames = kwargs.get('filename','')
        self.maskFile  = kwargs.get('maskfile','')
        self.varFile   = kwargs.get('varfile','')


        self.combine   = kwargs.get('combine','mean')

        if(self.combine=='mean'):
            self.dataCube = self.return_meancube()

        elif(self.combine=='median'):
            self.dataCube = self.return_mediancube()
            
        if(self.maskFile==''):
            self.maskData = np.array([])
        else:
            self.maskData = fits.getdata(self.maskFile)
        
        if(self.varFile==''):
            self.varData  = np.array([])
            self.errData  = np.array([])
        else:
            self.varData  = fits.getdata(self.varFile)
            self.errData  = np.sqrt(self.varData)


        self.meanStart    = kwargs.get('meanstart',500)
        self.meanEnd     = kwargs.get('meanend', 1500)

        self.dataCubeMean = np.mean(self.dataCube[self.meanStart:self.meanEnd,:,:],0)

        self.rscale, self.gscale, self.bscale     = kwargs.get('scale', [1,1.4,2])
        self.Q, self.alpha                        = kwargs.get('q_alpha', [3,0.4])
        self.masklevel, self.maskoffset           = kwargs.get('mask_offset', [-1.0,0.0])
        self.saturation, self.backsub, self.vb    = kwargs.get('saturation_backsub_vb',['white',False,False])


    def spectraUnderMask(self, method='sum'):
        """
        Computes the spectra under the mask. The method can be either sum or mean.

        Args:
            method (str, optional): Method to collapse mask spectra. Defaults to 'sum'.
        """
        y,x = np.where(self.maskData >0)
        dataStack = []
        for i in range(len(y)):
            dataStack.append(self.dataCube[:,y[i],x[i]])
        
        dataStack = np.asarray(dataStack)
        if(method=='sum'):
            dataSum = np.nansum(dataStack,axis=0)
            return(dataSum)
        
        elif(method=='mean'):
            dataMean = np.nanmean(dataStack,axis=0)        
            return(dataMean)
    
    def whiteLightImage(self,**kwargs):
        plt.imshow(
            self.dataCubeMean,
            origin    = 'lower',
            cmap      = kwargs.get('cmap'),
            vmin      = kwargs.get('vmin'),
            vmax      = kwargs.get('vmax')
        )

    def overlayMask(self,**kwargs):
        plt.imshow(
            self.maskData,
            origin    = 'lower',
            cmap      = kwargs.get('cmap'),
            vmin      = kwargs.get('vmin'),
            vmax      = kwargs.get('vmax'),
            alpha     = kwargs.get('alpha',0.2)
        )


    def return_mediancube(self):
        data_temp = []
        for i in self.fileNames:
            data_temp.append(fits.getdata(i))

        return(np.median(np.array(data_temp),axis=0))

    def return_meancube(self):
        data_temp = []
        for i in self.fileNames:
            data_temp.append(fits.getdata(i))
        return(np.mean(np.array(data_temp),axis=0))


    def humviPNG(self, bwave=[500,1000], gwave=[800,1200], rwave=[1200,1500]):
        """
        This function will create a png file using the humvi package.
        rwave, gwave, bwave are the wavelength ranges for the red, green, and blue channels.
        """
        bfile   = np.sum(self.dataCube[bwave[0]:bwave[1]],0)
        hdu     = fits.PrimaryHDU(data=bfile)
        hdu.writeto(f"{self.objid}_B.fits",overwrite=True)

        gfile   = np.sum(self.dataCube[gwave[0]:gwave[1]],0)
        hdu     = fits.PrimaryHDU(data=gfile)
        hdu.writeto(f"{self.objid}_G.fits",overwrite=True)

        rfile   = np.sum(self.dataCube[rwave[0]:rwave[1]],0)
        hdu     = fits.PrimaryHDU(data=rfile)
        hdu.writeto(f"{self.objid}_R.fits",overwrite=True)
        
        humvi.compose(
            f"{self.objid}_R.fits",
            f"{self.objid}_G.fits",
            f"{self.objid}_B.fits",
            scales     = (self.rscale, self.gscale, self.bscale),
            Q          = self.Q,
            alpha      = self.alpha,
            masklevel  = self.masklevel,
            saturation = self.saturation,
            offset     = self.maskoffset,
            backsub    = self.backsub,
            vb         = self.vb,
            outfile    = self.objid+"_cubePng.png"
        )

        display_image(self.objid+"_cubePng.png")
