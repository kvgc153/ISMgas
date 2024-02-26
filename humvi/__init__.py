#from io import *
#from pjm import *
#from lupton import *
#from compose import *

#__all__ = ["io", "pjm", "lupton","compose"]


import numpy
import sys,getopt
from PIL import Image
import humvi
import os
import astropy.io.fits as pyfits



# =====================================================================

def compose(rfile, gfile, bfile, scales=(1.0,1.0,1.0), Q=1.0, alpha=1.0, \
            masklevel=None, saturation='color', offset=0.0, backsub=False, \
            vb=False, outfile='color.png'):
    """
    Compose RGB color image.
    """

    # -------------------------------------------------------------------

    if vb:
        print("HumVI: Making color composite image of data in following files:",rfile,gfile,bfile)
        print("HumVI: Output will be written to",outfile)
        if masklevel is not None: print ("HumVI: Masking stretched pixel values less than",masklevel)

    # Read in images, calibrated into flux units:

    band3 = humvi.channel(rfile)
    band2 = humvi.channel(gfile)
    band1 = humvi.channel(bfile)

    # Check shapes are equal:
    humvi.check_image_shapes(band1.image,band2.image,band3.image)

    # Subtract backgrounds (median, optional):
    if backsub:
      band1.subtract_background()
      band2.subtract_background()
      band3.subtract_background()

    # -------------------------------------------------------------------

    # BUG: as it stands, this code assumes one file one channel, whereas
    # in practice we might like to be able to make composites based on
    # N bands. Need to overload + operator for channels? Calib etc will
    # need altering as well as image.

    red = band3
    green = band2
    blue = band1

    # -------------------------------------------------------------------
    # Set scales determining color balance in composite:

    rscale,gscale,bscale = humvi.normalize_scales(scales)
    red.set_scale(manually=rscale)
    green.set_scale(manually=gscale)
    blue.set_scale(manually=bscale)
    if vb:print ('HumVI: Scales normalized to:',red.scale,green.scale,blue.scale)

    # Scale images - only do once:
    red.apply_scale()
    green.apply_scale()
    blue.apply_scale()

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Stretch images to cope with high dynamic range:

    if vb:
        print("HumVI: Stretch parameters Q,alpha:",Q,alpha)
        print("HumVI: At low surface brightness levels, the channel images are further rescaled by alpha")
        print("HumVI: Nonlinearity sets in at about 1/Q*alpha in the scaled intensity image:",1.0/(Q*alpha))

    # Compute total intensity image and the arcsinh of it:
    I = humvi.lupton_intensity(red.image,green.image,blue.image,type='sum')
    stretch = humvi.lupton_stretch(I,Q,alpha)

    # Apply stretch to channel images:
    r = stretch * red.image
    g = stretch * green.image
    b = stretch * blue.image

    if masklevel is not None:
        # Mask problem areas - exact zeros or very negative patches should
        # be set to zero.

        # BUG: this should have been done after scaling but before conversion
        # to channels, as its the individual images that have problems...

        r,g,b = humvi.pjm_mask(r,g,b,masklevel)

    # Offset the stretched images to make zero level appear dark gray.
    # Negative offset makes background more black...
    r,g,b = humvi.pjm_offset(r,g,b,offset)

    if saturation == 'color':
        # Saturate to colour at some level - might as well be 1, since
        # Q redefines scale?:
        threshold = 1.0
        r,g,b = humvi.lupton_saturate(r,g,b,threshold)
    # Otherwise, saturate to white.

    # Package into a python Image, and write out to file:
    image = humvi.pack_up(r,g,b)
    image.save(outfile)

    if vb: print( "HumVI: Image saved to:",outfile)

    return

# ======================================================================

vb = 0

# ======================================================================

# BUG: This code implies one file one channel, whereas we want to make
# composites based on N images... Class should be image, not channel.
# RGB channels should be constructed *after* scaling but *before* stretching

class channel:

    def __init__(self,fitsfile):

        self.input = fitsfile
        # Read in image and header:
        hdulist = pyfits.open(self.input)
        # self.hdr = hdulist[0].header
        # self.image = hdulist[0].data
        # Picking -1 header assumes we have 1 extension or PS1 (2 ext, image is last)
        self.image = hdulist[-1].data
        self.hdr = hdulist[-1].header
        self.calibrate()
        hdulist.close()

        return

    def calibrate(self):

        # Which telescope took these data?
        self.get_origin()

        # Get zero point, exptime:
        self.get_zeropoint()
        # EXPTIME is 1.0 for images in counts/s - but headers do not always
        # get this right... get_exptime gets this right.
        self.get_exptime()
        # Airmass? Gain? Should be included in zeropoint.

        # print "Image statistics for "+self.input
        # print "  ",self.origin,self.exptime,self.zpt

        # # Report 5 sigma depth:
        # image = self.image.copy()
        # mean = numpy.average(image)
        # stdev = numpy.std(image)
        # nsigma = 1
        # clip = 3
        # # Apply clipping:
        # while nsigma > 0.01:
        #     index = numpy.where(abs((self.image - mean)/stdev) < clip)[0]
        #     image = image[index]
        #     newmean = numpy.average(image)
        #     newstdev = numpy.std(image)
        #     nsigma = abs(mean - newmean)/newstdev
        #     mean = newmean
        #     stdev = newstdev
        # print "  Before calibration, mean, rms = ",mean,stdev
        # depth = -2.5*numpy.log10(5.0*stdev) + self.zpt
        # print "  Approximate 5-sigma limiting magnitude: ",depth

        # Compute calibration factor for image pixel values to
        # convert them into flux units. The 30 is arbitrary, and
        # simply determines the absolute value of alpha required
        # for a nice image.
        self.calib = (10.0**(0.4*(30.0 - self.zpt))) / self.exptime
        self.image *= self.calib

        # # Report 5 sigma depth:
        # image = self.image.copy()
        # mean = numpy.average(image)
        # stdev = numpy.std(image)
        # nsigma = 1
        # clip = 3
        # # Apply clipping:
        # while nsigma > 0.01:
        #     index = numpy.where(abs((self.image - mean)/stdev) < clip)[0]
        #     image = image[index]
        #     newmean = numpy.average(image)
        #     newstdev = numpy.std(image)
        #     nsigma = abs(mean - newmean)/newstdev
        #     mean = newmean
        #     stdev = newstdev
        # print "  After calibration, mean, rms = ",mean,stdev

        return

    def get_origin(self):
        if 'TELESCOP' in self.hdr:
            if self.hdr['TELESCOP'] == 'CFHT 3.6m':
                self.origin = 'CFHT'
            elif self.hdr['TELESCOP'] == 'ESO-VLT-U0':
                # Assume that all data from ESO-VLT-U0 is from KIDS.
                self.origin = "KIDS"
            else:
                self.origin = self.hdr['TELESCOP']
        elif 'ORIGIN' in self.hdr:
            if self.hdr['ORIGIN'] == 'CFHT':
                self.origin = 'CFHT'
            elif self.hdr['ORIGIN'] == 'DES':
                self.origin = 'DES'
            else:
                self.origin = self.hdr['ORIGIN']
        elif 'PSCAMERA' in self.hdr:
            self.origin = 'PS1'
        elif 'FID_ZP' in self.hdr:
            self.origin = 'DES'
        elif 'PROV' in self.hdr:
            self.origin = 'VICS82'
        else:
            self.origin = 'UNKNOWN'
        return

    def get_zeropoint(self):
        if self.origin == 'CFHT':
            if 'MZP_AB' in self.hdr:
                self.zpt = self.hdr['MZP_AB']
            elif 'MAGZP' in self.hdr:
                self.zpt = self.hdr['MAGZP']
            # elif 'PHOT_C' in self.hdr:
            #     self.zpt = self.hdr['PHOT_C']
            else:
                self.zpt = 30.0
        elif self.origin == 'PS1':
            self.zpt = self.hdr['HIERARCH FPA.ZP']
        elif self.origin == 'DES':
            self.zpt = self.hdr['MZP_AB']
        elif self.origin == 'VICS82':
            if 'MZP_AB' in self.hdr:
                self.zpt = self.hdr['MZP_AB']
            else:
                self.zpt = 30.0
        elif self.origin == 'KIDS':
            # KiDS coadds are calibrated to ZPT=0.
            self.zpt = 0.0
        else: # UNKNOWN
            self.zpt = 30.0
        return

    def get_exptime(self):
        # Here we assume that both CFHT and PS1 provide images with
        # pixel values in counts per second... or that the zero point
        # takes into account the exptime.
        if self.origin == 'CFHT':
            self.exptime = 1.0
        elif self.origin == 'PS1':
            # self.exptime = self.hdr['EXPTIME']
            self.exptime = 1.0
        elif self.origin == 'DES':
            # self.exptime = self.hdr['EXPTIME']
            self.exptime = 1.0
        elif self.origin == 'VICS82':
            # self.exptime = self.hdr['EXPTIME']
            self.exptime = 1.0
        elif self.origin == 'KIDS':
            # self.exptime = self.hdr['EXPTIME']
            self.exptime = 1.0
        else: #UNKNOWN
            # Use 1.0 as default to ensure the program doesn't crash.
            self.exptime = 1.0
        return

    def set_scale(self,manually=False):
        if manually:
             self.scale = manually
        else:
             self.scale = 1.0
        return

    def apply_scale(self):
        self.image *= self.scale
        return

    def subtract_background(self):
        self.image -= numpy.median(self.image)
        return

    def writefits(self):
        self.output = str.split(self.input,'.')[0]+'_calibrated.fits'
        if os.path.exists(self.output): os.remove(self.output)
        hdu = pyfits.PrimaryHDU()
        hdu.header = self.hdr
        hdu.data = self.image
        hdu.verify()
        hdu.writeto(self.output)
        return

# ======================================================================

def normalize_scales(scales):
    assert len(scales) == 3
    s1,s2,s3 = scales
    mean = (s1 + s2 + s3)/3.0
    return s1/mean, s2/mean, s3/mean

# ----------------------------------------------------------------------

def filter2wavelength(fname):

    # CFHT MegaCam (from http://www.cfht.hawaii.edu/Instruments/Imaging/Megacam/specsinformation.html)
    if fname == 'u.MP9301':
        L = 3740
    elif fname == 'g.MP9401':
        L = 4870
    elif fname == 'r.MP9601':
        L = 6250
    elif fname == 'i.MP9701' or 'i.MP9702':
        L = 7700
    elif fname == 'z.MP9801':
        L = 9000

    # SDSS:

    # DES:

    # etc
    return L

# ----------------------------------------------------------------------

def check_image_shapes(r,g,b):

    if (numpy.shape(r) != numpy.shape(g)) or \
        (numpy.shape(r) != numpy.shape(b)):
        raise "Image arrays are of different shapes, exiting"

    return

# ----------------------------------------------------------------------
# Make an 8 bit integer image cube from three channels:

def pack_up(r,g,b):

    NX,NY = numpy.shape(r)

    x = numpy.zeros([NX,NY,3])
    x[:,:,0] = numpy.flipud(r)
    x[:,:,1] = numpy.flipud(g)
    x[:,:,2] = numpy.flipud(b)

    x = numpy.clip(x,0.0,1.0)
    x = x*255

    return Image.fromarray(x.astype(numpy.uint8))

# ======================================================================


# ======================================================================

def lupton_intensity(r,g,b,type='sum'):

    if type == 'sum':
        return (r+g+b) + 1e-10

    elif type == 'rms':
        return numpy.sqrt(r*r+g*g+b*b) + 1e-10

# ----------------------------------------------------------------------

def lupton_stretch(I,Q,alpha):

    return numpy.arcsinh(alpha*Q*I) / (Q*I)

# ----------------------------------------------------------------------
# Clip high values to box:

def lupton_saturate(r,g,b,threshold):

    x = numpy.dstack((r,g,b))

    # Highest pixel-value at given position
    maxpix = numpy.max(x, axis=-1)
    maxpix[maxpix<1.0] = 1.0

    rr = r/maxpix
    gg = g/maxpix
    bb = b/maxpix

    return rr,gg,bb

# ======================================================================
# Testing:

if __name__ == '__main__':

    print("No tests defined")

# ======================================================================






# ======================================================================

"""
Functions for implementing Phil's tweaks to the Lupton algorithm for
making color composite images.
"""

# ======================================================================
# Globally useful modules:

import numpy
from PIL import Image

# ======================================================================
# Add small offset to image, to make background look dark gray not black:

def pjm_offset(r,g,b,offset):

    rr = r + offset
    gg = g + offset
    bb = b + offset

    return rr,gg,bb

# ----------------------------------------------------------------------
# Detect problem areas in any of the channel images, and mask out:

def pjm_mask(r,g,b,threshold):

    tiny = 1e-10
    mask = r*0.0 + 1.0

    for image in (r,g,b):

        image[numpy.isnan(image)] = 0.0
        image[numpy.isinf(image)] = 0.0

        mask[image < threshold] = 0.0
        mask[(image > -tiny) & (image < tiny)] = 0.0

    return r*mask,g*mask,b*mask

# ======================================================================

