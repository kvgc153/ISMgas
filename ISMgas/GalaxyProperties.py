import requests
from sparcl.client import SparclClient
from astropy.convolution import convolve, Gaussian1DKernel

from humvi import compose 

from ISMgas.linelist import linelist_SDSS
from ISMgas.SupportingFunctions import *

class GalaxyProperties:
    def __init__(self,**kwargs):
        self.objid            = kwargs.get('objid')
        self.ra               = kwargs.get('ra')            ## ra in degrees
        self.dec              = kwargs.get('dec')           ## dec in degrees
        self.zs               = kwargs.get('zs')            ## redshift of the source galaxy
        self.zsNotes          = kwargs.get('zsNotes')
        self.zdef             = kwargs.get('zdef',[])       ## redshift of the deflector galaxy if known
        self.zdefNotes        = kwargs.get('zdefNotes')
        self.inst             = kwargs.get('inst')          ## Instrument used to observe target e.g ESI/NIRES/etc
        self.inst_pipeline    = kwargs.get('inst_pipeline') ## pipeline used to reduce the data
        self.inst_sigma       = kwargs.get('inst_sigma',1)  ## Instrument resolution sigma in km/s
        self.spec_filename    = kwargs.get('spec_filename') ## filename containing the fits file
        self.mass             = kwargs.get('mass',[])
        self.sfr              = kwargs.get('sfr',[])
        self.survey           = kwargs.get('survey')
        self.merger           = kwargs.get('merger')
        self.mergerNotes      = kwargs.get('mergerNotes')
        self.R                = kwargs.get('R')

    ##################
    ### kvgc files ###
    ##################

    def decalsFitsAndPng(self, grab=False, **kwargs):
        '''
        kwargs:

        'grab': Grab the fits file from server or use the one that exists. False by default
        'ra': RA
        'dec': DEC
        'pixscale': default is 0.257
        'layer': default is ls-dr9 (could be hsc-dr2)
        'size': Size of output image

        humvi params:

        rscale, gscale, bscale    = kwargs.get('scale', [1,1.5,3])
        Q,alpha                   = kwargs.get('q,alpha', [1,15])
        masklevel,offset          = kwargs.get('mask,offset', [-1.0,0.0])
        saturation,backsub,vb     = kwargs.get('saturation,backsub,vb',['white',False,False])

        '''
        ## grab fits if needed ##
        if(grab==True):

            r = requests.get(
                "https://www.legacysurvey.org/viewer/cutout.fits?",
                params = {
                    'ra'          : kwargs.get('ra',self.ra),
                    'dec'         : kwargs.get('dec',self.dec),
                    'pixscale'    : kwargs.get('pixscale',0.257),
                    'layer'       : kwargs.get('layer','ls-dr9'),
                    'size'        : kwargs.get('size', 100)
                    },
                )
            with open("%s.fits"%(self.objid),'wb') as foo_fits:
                foo_fits.write(r.content)

            filename    = "%s.fits"%(self.objid)
            data        = fits.getdata(filename)
            hdr         = fits.getheader(filename)

            ## split into separate bands
            count=0
            for i in [hdr['BAND0'],hdr['BAND1'],hdr['BAND2']]:
                hdu         = fits.PrimaryHDU()
                hdu.data    = data[count]
                hdu.header  = hdr
                hdu.writeto("%s_%s.fits"%(self.objid,i),overwrite=True)
                count       += 1


        ## Make png image from decals image ##
        rscale, gscale, bscale    = kwargs.get('scale', [1,1.5,3])
        Q,alpha                   = kwargs.get('q_alpha', [1,15])
        masklevel,offset          = kwargs.get('mask_offset', [-1.0,0.0])
        saturation,backsub,vb     = kwargs.get('saturation_backsub_vb',['white',False,False])

        rfile                     = "%s_%s.fits"%(self.objid,fits.getheader("%s.fits"%(self.objid))['BAND2'])
        gfile                     = "%s_%s.fits"%(self.objid,fits.getheader("%s.fits"%(self.objid))['BAND1'])
        bfile                     = "%s_%s.fits"%(self.objid,fits.getheader("%s.fits"%(self.objid))['BAND0'])

        outfile                   = "%s.png"%(self.objid)
        print(rfile, gfile, bfile,(rscale,gscale,bscale), Q, alpha, masklevel, saturation, offset, backsub, vb, outfile)

        compose(
            rfile,
            gfile,
            bfile,
            scales        = (rscale,gscale,bscale),
            Q             = Q,
            alpha         = alpha,
            masklevel     = masklevel,
            saturation    = saturation,
            offset        = offset,
            backsub       = backsub,
            vb            = vb,
            outfile       = outfile
        )

    def fitscgi_HST(self, **kwargs):
        getVars = {
                'ra'              : self.ra,
                'dec'             : self.dec,
                'blue'            : kwargs.get('blue',''),
                'green'           : kwargs.get('green',''),
                'red'             : kwargs.get('red',''),
                'zoom'            : kwargs.get('zoom',1),
                'size'            : kwargs.get('size',1400),
                'Asinh'           : kwargs.get('asinh',True),
                'download'        : kwargs.get('download',True),
                'Align'           : kwargs.get('align',True),
                'autoscale'       : kwargs.get('autoscale',''),
                'AutoscaleMax'    : kwargs.get('autoscalemax',''),
                'AutoscaleMin'    : kwargs.get('autoscalemin',''),
                'compass'         : kwargs.get('compass',True ),
                'format'          : kwargs.get('format', 'jpg')
        }

        foo_HST = requests.get(
                "https://hla.stsci.edu/cgi-bin/fitscut.cgi?",
                params = getVars
        )
        
        if(getVars['format'] == 'jpg'):
            with open("%s_HST_fitscgi.jpg"%(self.objid),'wb') as foo_HST_png:
                foo_HST_png.write(foo_HST.content)

        elif(getVars['format'] == 'fits'):
            with open("%s_HST_fitscgi.fits"%(self.objid),'wb') as foo_HST_fits:
                foo_HST_fits.write(foo_HST.content)


    def SDSS_spectra(self,index=0,search_min=0.0005, search_max=0.0005, ylim = [-1,5]):
        '''
        Adapted from this example from the datalab: https://github.com/astro-datalab/notebooks-latest/blob/master/04_HowTos/SPARCL/How_to_use_SPARCL.ipynb
        '''
        client                    = SparclClient()
        out                       = ['id', 'specid','ra', 'dec', 'redsstem christie.hift', 'spectype', 'data_release', 'redshift_err']
        cons = {
                'data_release'    : ['BOSS-DR16', 'SDSS-DR16'],
                'ra'              : [self.ra-search_min,self.ra+search_max],
                'dec'             : [self.dec-search_min,self.dec+search_max]
            }


        found_I                   = client.find(outfields=out, constraints=cons)
        # print(["%s, %f, %f, %f, %f" % (f['id'],f['ra'],f['dec'],f['redshift'],f['redshift_err']) for f in found_I.records])

        # Define the fields to include in the retrieve function
        inc           = ['specid', 'data_release', 'redshift', 'redshift_err', 'flux', 'wavelength', 'model', 'ivar', 'mask', 'spectype']
        results_II    = client.retrieve_by_specid(
                                                    specid_list       = [f['specid'] for f in found_I.records],
                                                    include           = inc,
                                                    dataset_list    = ['SDSS-DR16','BOSS-DR16']
                                            )

        self.plot_SDSS_spec(index = index,results = results_II, ylim = ylim)

    def plot_SDSS_spec(self, index, results, ylim=[-1,5]):
        """
        Pass an index value and the output from using client.retrieve()
        to plot the spectrum at the specified index.
        """

        record                  = results.records[index]

        # id_                     = record.id
        # self.SDSS_id            = id_

        data_release            = record.data_release
        flux                    = record.flux
        self.SDSS_flux          = flux

        wavelength              = record.wavelength
        self.SDSS_wavelength    = wavelength

        model                   = record.model
        self.SDSS_model         = model

        spectype                = record.spectype
        self.SDSS_spectype      = spectype

        redshift                = record.redshift
        self.SDSS_redshift      = redshift
        self.SDSS_redshift_err  = record.redshift_err

        plt.title(f"Data Set = {data_release}, spectype = {spectype}, z={redshift}$\pm${self.SDSS_redshift_err}")
        plt.xlabel('$\lambda\ [\AA]$')
        plt.ylabel('$f_{\lambda}$ $(10^{-17}$ $erg$ $s^{-1}$ $cm^{-2}$ $\AA^{-1})$')

        # Plot unsmoothed spectrum in grey
        plt.plot(wavelength, flux, color='k', alpha=0.2, label='Unsmoothed spectrum')

        # Overplot spectrum smoothed using a 1-D Gaussian Kernel in black
        plt.plot(wavelength, convolve(flux, Gaussian1DKernel(5)), color='k', label='Smoothed spectrum')

        # Overplot the model spectrum in red
        plt.plot(wavelength, model, color='r', label='Model spectrum')

        lines_keys = linelist_SDSS.keys()
        for i in lines_keys:
            if(
                linelist_SDSS[i]['lambda']*(1+self.SDSS_redshift)>4000 and 
                linelist_SDSS[i]['lambda']*(1+self.SDSS_redshift)<10000 and 
                linelist_SDSS[i]['plot']==True
            ):
                plt.axvline(
                    linelist_SDSS[i]['lambda']*(1+self.SDSS_redshift), 
                    0.95, 
                    1.0, 
                    color = 'b', 
                    lw = 3.0
                )
                plt.axvline(
                    linelist_SDSS[i]['lambda']*(1+self.SDSS_redshift), 
                    color = 'b', 
                    lw = 1.0, 
                    linestyle = ':'
                )
                trans = plt.gca().get_xaxis_transform()
                plt.gca().annotate(
                            i,
                            xy          = (linelist_SDSS[i]['lambda']*(1+self.SDSS_redshift), 1.1),
                            xycoords    = trans,
                            fontsize    = 10,
                            rotation    = 90,
                            color       = 'b'
                            )

        plt.legend()
        plt.ylim(ylim)

        ## Save 
        plt.tight_layout()
        plt.savefig(f"{self.objid}-SDSS_spectra.png")
