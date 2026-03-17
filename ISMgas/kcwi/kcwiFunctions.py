import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.constants import c
c_kms = c*1e-3

import matplotlib.pyplot as plt
import humvi

import astropy.units as u

## Custom packages
from ISMgas.fitting.DoubleGaussian import *
# from ISMgas.GalaxyProperties import GalaxyProperties
from ISMgas.linelist import linelist_highz,linelist_SDSS
from ISMgas.SupportingFunctions import display_image

class kcwiAnalysis():

    def __init__(self,**kwargs):

        # GalaxyProperties.__init__(self,**kwargs)

        self.fileName = kwargs.get('filename','')
        self.maskFile  = kwargs.get('maskfile','')
        self.varFile   = kwargs.get('varfile','')
        self.objid    = kwargs.get('objid','')

        self.hdr = fits.getheader(self.fileName)
        self.dataCube = fits.getdata(self.fileName)
        try:
            self.wave    = (np.arange(self.hdr['NAXIS3']) - self.hdr['CRPIX3'] + 1) * self.hdr['CDELT3'] + self.hdr['CRVAL3']
        except KeyError:
            self.wave    = (np.arange(self.hdr['NAXIS3']) - self.hdr['CRPIX3'] + 1) * self.hdr['CD3_3'] + self.hdr['CRVAL3']

        if(self.hdr['CUNIT3'] == 'Angstrom'):
            self.wave = self.wave * u.AA
        # self.combine   = kwargs.get('combine','mean')

        # if(self.combine=='mean'):
        #     self.dataCube = self.return_meancube()

        # elif(self.combine=='median'):
        #     self.dataCube = self.return_mediancube()
        
        self.specMask = []
        if(self.maskFile==''):
            self.maskData = np.array([])
        else:
            self.maskData = fits.getdata(self.maskFile)

            y,x = np.where(self.maskData >0)
            dataStack = []
            for i in range(len(y)):
                dataStack.append(self.dataCube[:,y[i],x[i]])
            
            self.specMask = np.asarray(dataStack)

        if(self.varFile==''):
            self.varData  = np.array([])
            self.errData  = np.array([])
        else:
            self.varData  = fits.getdata(self.varFile)
            self.errData  = np.sqrt(self.varData)


        self.meanStart    = kwargs.get('meanstart',500)
        self.meanEnd     = kwargs.get('meanend', 1500)

        try:
            self.dataCubeMean = np.mean(self.dataCube[self.meanStart:self.meanEnd,:,:],0)
        except IndexError:
            ## if there is only frame in the datacube, use that frame as the mean
            self.dataCubeMean = self.dataCube 
        self.rscale, self.gscale, self.bscale     = kwargs.get('scale', [1,1.4,2])
        self.Q, self.alpha                        = kwargs.get('q_alpha', [3,0.4])
        self.masklevel, self.maskoffset           = kwargs.get('mask_offset', [-1.0,0.0])
        self.saturation, self.backsub, self.vb    = kwargs.get('saturation_backsub_vb',['white',False,False])


    def spectraUnderMask(self, method='sum', plot=False):
        """
        Computes the spectra under the mask. The method can be either sum or mean.

        Args:
            method (str, optional): Method to collapse mask spectra. Defaults to 'sum'.
        """

        if(method=='sum'):
            dataSum = np.nansum(self.specMask,axis=0)
            plt.plot(self.wave, dataSum, color='black', drawstyle='steps-mid')
            return(dataSum)
        
        elif(method=='mean'):
            dataMean = np.nanmean(self.specMask,axis=0)      
            plt.plot(self.wave, dataMean        , color='black', drawstyle='steps-mid')  
            return(dataMean)
        
        elif(method=='median'):
            dataMedian = np.nanmedian(self.specMask,axis=0)    
            plt.plot(self.wave, dataMedian, color='black', drawstyle='steps-mid')    
            return(dataMedian)
    

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

    def humviPNG(self, bwave=[500,1000], gwave=[800,1200], rwave=[1200,1500]):
        """
        This function will create a png file using the humvi package.
        rwave, gwave, bwave are the wavelength ranges for the red, green, and blue channels.
        """
        bfile   = np.nansum(self.dataCube[bwave[0]:bwave[1]],0)
        hdu     = fits.PrimaryHDU(data=bfile)
        hdu.writeto(f"{self.objid}_B_humvi.fits",overwrite=True)

        gfile   = np.nansum(self.dataCube[gwave[0]:gwave[1]],0)
        hdu     = fits.PrimaryHDU(data=gfile)
        hdu.writeto(f"{self.objid}_G_humvi.fits",overwrite=True)

        rfile   = np.nansum(self.dataCube[rwave[0]:rwave[1]],0)
        hdu     = fits.PrimaryHDU(data=rfile)
        hdu.writeto(f"{self.objid}_R_humvi.fits",overwrite=True)
        
        humvi.compose(
            f"{self.objid}_R_humvi.fits",
            f"{self.objid}_G_humvi.fits",
            f"{self.objid}_B_humvi.fits",
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
