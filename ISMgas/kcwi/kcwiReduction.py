from ISMgas.kcwi.kcwiFunctions import *
from ISMgas.visualization.fits import ScaleImage

class kcwiReduxMontage:
    def __init__(self, galaxy, filenames, pixscale=0.3, **kwargs):
        self.ra = galaxy.ra
        self.dec = galaxy.dec
        self.objid = galaxy.objid
        self.filenames = filenames
        self.pixscale = pixscale ## arcseconds
        self.dataCube  = None
        
              
    def combineAndDrizzle(self, drizzleFactor=0.7):
        ### Write a DECaLS like header
        template = f"""
SIMPLE  =                    T / file does conform to FITS standard
BITPIX  =                  -32 / number of bits per data pixel
NAXIS   =                    2 / number of data axes
NAXIS1  =                  140 / length of data axis 1
NAXIS2  =                  140 / length of data axis 2
EXTEND  =                    T / FITS dataset may contain extensions
COMMENT   FITS (Flexible Image Transport System) format is defined in 'Astronomy
COMMENT   and Astrophysics', volume 376, page 359; bibcode: 2001A&A...376..359H
SURVEY  = 'LegacySurvey'
VERSION = 'DR9     '
IMAGETYP= 'IMAGE   '           / None
CTYPE1  = 'RA---TAN'           / TANgent plane
CTYPE2  = 'DEC--TAN'           / TANgent plane
CRVAL1  =              {self.ra} / Reference RA
CRVAL2  =             {self.dec} / Reference Dec
CRPIX1  =                 70.5 / Reference x
CRPIX2  =                 70.5 / Reference y
CD1_1   = -{self.pixscale/3600} / CD matrix
CD1_2   =                   0. / CD matrix
CD2_1   =                   0. / CD matrix
CD2_2   = {self.pixscale/3600} / CD matrix
END
"""
        f = open(f"{self.objid}.hdr", 'w+')
        f.write(template)
        f.close()        
        
        dd = kcwiAnalysis(
            filename = self.filenames,
        )

        fits.writeto(
            filename = self.objid + ".fits",
            data = dd.dataCube,
            header = fits.getheader(self.filenames[0]),
            overwrite= True
        )          

        ## Project datacube using montage
        cmd = f"mProjectCube -X -z 0.7 {self.objid}.fits  {self.objid}_drizzle.fits  {self.objid}.hdr"
        runCMD(cmd)
        print(f"Drizzled file: {self.objid}_drizzle.fits")
        
        ## Show user the drizzled datacube
        self.dataCube = fits.getdata(f"{self.objid}_drizzle.fits")
        dd = kcwiAnalysis(
            objid = 'test',
            filename = [f"{self.objid}_drizzle.fits"],
        )
        print(f"Shape: {np.shape(dd.dataCube)}")

        ScaleImage(dd.dataCubeMean).plot()
        

def padAndAlign(cubes, newOutputShape, centroids=[]):
    
    results = []
    alignStack = []
        
    for i in range(len(cubes)):
        
        newCube = np.zeros(newOutputShape)
        newCube[:,0:cubes[i].dataCube.shape[1], 0:cubes[i].dataCube.shape[2]] = cubes[i].dataCube

        
        if(len(centroids)==0):
            results.append(newCube)
            alignStack.append(np.mean(newCube[500:-500, :, :], axis=0))
            
        else:
            newAlignCube = np.roll(
                newCube, 
                [0, -(centroids[i][0] - centroids[0][0]), -(centroids[i][1]- centroids[0][1])],
                axis=(0,2,1)
            )
            results.append(newAlignCube)
            alignStack.append(np.mean(newAlignCube[500:-500, :, :], axis=0))
        
    fits.writeto(
        filename = 'align.fits',
        data = np.array(alignStack),
        overwrite= True
    )
    print("Use align.fits to manually align the datacubes")
    return(np.mean(results,axis=0))
    