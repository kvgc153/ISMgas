'''
Class handling i/o of spectra and making a velocity profile.
This code has been rigourously tested with data obtained from ESI and MagE. 
Feel free to try it out with other instruments. If you run into any issues, 
please report it on the repo                                                
'''

## Packages for plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

## Astropy
from astropy.io import fits
from astropy.table import Table

## scipy
from scipy.ndimage import gaussian_filter, gaussian_filter1d

## Custom packages
from ISMgas.fitting.DoubleGaussian import *
from ISMgas.GalaxyProperties import *
from ISMgas.linelist import linelist_highz
from ISMgas.globalVars import *
from ISMgas.SupportingFunctions import beautifyPlot
 

class AnalyzeSpectra(GalaxyProperties):
    '''
    Class handing the analysis of 1D spectra.  
    Example usage :
    ```
    obj = AnalyzeSpectra(
        objid           = "stack",
        spec_filename   = "stack.fits",
        R               = 4000,
        zs              = 0
    )


    obj.combineISMLines(
        Nsm             = 1,
        chosen_lines    = ['Si II 1260', 'Si II 1526'],
        wavrange        = [
            [-2000,2000],
            [-2000,2000]
            
        ],
    )
    foo = load_pickle('stack-combined.pkls')
    plotWithError(
        foo['source_wav'],
        foo['source_flux'],
        foo['source_error']
    )

    plt.ylim([0,1.3])


    ```
    '''

    def __init__(self, **kwargs):

        GalaxyProperties.__init__(self,**kwargs)
        
        self.color_pal                    = color_pal_kvgc['pub1']
        self.combined_wavelength          = []
        self.combined_flux                = []
        self.combined_fluxerr             = []
        self.all_lines                    = []
        self.all_lines_err                = []
        self.line_names                   = []
        self.user_wavrange                = []

        self.user_chosen_lines            = []
        self.user_chosen_lines_err        = []
        self.user_chosen_lines_name       = []
        self.user_chosen_lines_wavrange   = []

        self.Nsm                          = 0
        self.deltaWavelength              = 25 ## in \AA

    def fetchData(self):
        hdu   = fits.open(self.spec_filename)
        t     = Table(hdu[1].data)
        wl    = np.array(t['LAMBDA'])[0]
        spec  = np.array(t['SPEC'])[0]
        ivar  = np.array(t['IVAR'])[0]
        
        return(wl, spec, ivar)
      
    def dataloader(
            self,
            Nsm,
            state,
            wavrange           = [],
            plotting           = False,
            chosen_lines       = 0,
            start_interp       = -2000,
            end_interp         = 2000,
            delta_interp       = 25,
            xlim               = [-2000,2000],
            ylim               = [0,2],
            interpolate_spec   = True 
        ):
        
        '''
        Handles importing data from pre-processed fits file. 
        returns (wavelength, flux, sigma, line_name)
        '''

        ### Make a dedicated user linelist or use linelist.py 
        if chosen_lines == 0:
            chosen_lines = linelist_highz.keys()

        count             = 1
        wav               = []
        spectra_interp    = []
        line_name         = []
        sigma             = []     

        if plotting:
            plt.figure(figsize = [13,18])

        for q in chosen_lines:
            linename   = q
            line_wl    = linelist_highz[q]["lambda"]

            if linelist_highz[q].get(state) is not None:
                wl, spec, ivar = self.fetchData()

                ## Choosing wavelengths +-deltaWavelength A from the absorption line 
                start   = (1+self.zs)*(line_wl - self.deltaWavelength)
                end     = (1+self.zs)*(line_wl + self.deltaWavelength)

                mask_start    = wl > start
                mask_end      = wl < end
                mask_wl       = mask_start & mask_end
                
                wavelength      = wl[mask_wl]
                flux            = spec[mask_wl]
                invvar          = ivar[mask_wl]             
     
                wavelength_rest = wavelength/(1+self.zs)

                ## Sanity check to ensure that the flux array has elements in it
                if(len(flux)!=0):
                    
                    ## Take weighted mean of Nsm elements
                    f_wtd, ww_wtd, sigma_wtd = returnWeightedArray(
                        Nsm          = Nsm,
                        spec         = flux,
                        ivar         = invvar,
                        wave_rest    = c_kms*(wavelength_rest-line_wl)/line_wl  ## changing from angstrom to km/s
                    )
                    
                    #*********************************************#
                    if(plotting):
                        plt.subplot(8,4,count)
                        plt.plot(
                            ww_wtd, f_wtd,
                            color=self.color_pal[14],
                            alpha = 0.8,
                            linewidth=3,
                            label='%s-%d'%(linename,line_wl)
                        )

                        if(len(wavrange)>0):
                            plt.axvline(
                                [wavrange[count-1][0]],
                                color='black',
                                linewidth=2,
                                linestyle='--'
                            )
                            plt.axvline(
                                [wavrange[count-1][1]],
                                color='black',
                                linewidth=2,
                                linestyle='--'
                            )

                        elif(len(wavrange)==0):
                            plt.text(
                                1300,0.3,
                                count,
                                fontsize=50,
                                alpha=0.4,
                                horizontalalignment='center',
                                verticalalignment='center'
                            )

                        font_size=15
                        # print("setting title to " + linename)
                        plt.title(linename,fontsize=font_size)
                        plt.xlabel('Velocity(km/s)',fontsize=font_size)
                        plt.ylabel('Flux',fontsize=font_size)
                        plt.xlim(xlim)
                        plt.ylim(ylim)
                        plt.xticks(np.arange(xlim[0],xlim[1]+1,1000),fontsize=12)
                        plt.yticks(np.arange(ylim[0],ylim[1],0.5),fontsize=12)
                        plt.grid('on',alpha=0.6)
                        plt.tight_layout()
                   #*********************************************#

                    count+=1

                    ## Begin Interpolation ##
                    if(interpolate_spec):
                        wav_interp    = np.arange(start_interp, end_interp, delta_interp)
                        wav.append(wav_interp)
                        
                        spectra_interp.append(
                            interpolateData(
                                    x       = ww_wtd,
                                    y       = f_wtd,
                                    xnew    = wav_interp
                            )
                        )
                        sigma.append(
                            interpolateData(
                                x       = ww_wtd,
                                y       = sigma_wtd,
                                xnew    = wav_interp
                            )
                        )
                        line_name.append(linename)
                    
                    else:
                        wav.append(ww_wtd)
                        spectra_interp.append(f_wtd)
                        sigma.append(sigma_wtd)
                        
                    ## End interpolation ##

        if plotting:
            fooPlotDict = {
                'lowion'    : {'title':"Low ionization ISM Lines", "saveFile": "%s-all-lines-lowions.png"%(self.objid)},
                'highion'   : {'title':"High ionization ISM Lines", "saveFile": "%s-all-lines-highions.png"%(self.objid)},
                'stellar'   : {'title':"Stellar Lines", "saveFile": "%s-stellar-lines.png"%(self.objid)},
                'opthin'    : {'title':"Optically thin Lines", "saveFile": "%s-opthin-lines.png"%(self.objid)},
                'fineem'    : {'title':"Fine structure Lines", "saveFile": "%s-fineem-lines.png"%(self.objid)},
                'nebem'     : {'title':"Nebular emission Lines", "saveFile": "%s-nebem-lines.png"%(self.objid)}
            }
            plt.suptitle(fooPlotDict[state]['title'],fontsize =25, y = 1.05)
            plt.savefig(fooPlotDict[state]['saveFile'],**ISMgasPlot['savefig'])

            plt.close()     
                           
        return(np.array(wav), np.array(spectra_interp), np.array(sigma), line_name)

    def plotLines(self, Nsm, xlim=[-2000,2000], ylim=[0,2], state='lowion'):
        '''
        Plots the individual ISM lines from spectra. Plots the raw spectra without interpolation.
        '''

        self.dataloader (
            Nsm                 = Nsm,
            state               = state,
            xlim                = xlim,
            ylim                = ylim,
            plotting            = True,
            interpolate_spec    = False,
        )

    def combineISMLines(
            self,
            Nsm,
            chosen_lines,
            wavrange        = [],
            xlim            = [-2000,2000],
            ylim               = [0,2],
            start_interp    = -2000,
            end_interp      = 2000,
            delta_interp    = 25,
            state           = 'lowion',
            normScaleFactor = 1,
            roi             = [] ):
        '''
            returns a weighted average of all the lines chosen by the user
        '''

        wav, spectra_interp, sigma, line_name = self.dataloader(
            Nsm            = Nsm,
            chosen_lines   = chosen_lines,
            start_interp   = start_interp,
            end_interp     = end_interp,
            delta_interp   = delta_interp,
            wavrange       = wavrange,
            state          = state,
            plotting       = True
        )
   
        # Normalize all spectra using median (RUN THIS ONLY ONCE!)
        for i in reversed(range(len(spectra_interp))):
            median_val = np.median(spectra_interp[i])

            # Multiply by normscalefactor (Default normScaleFactor=1) so this step does not do anything
            # But this is useful when the flux is not normalized to 1 but needs manual tweaking...
            median_val *= normScaleFactor

            # Divide both the spectra and the sigma by this median value
            spectra_interp[i]   = spectra_interp[i]/median_val
            sigma[i]            = sigma[i]/median_val
        
        ##-- Save --##
        allLinesSpec = {
            'wav'               : wav,
            'spectra_interp'    : spectra_interp,
            'sigma'             : sigma,
            'line_name'         : line_name,
            'wavrange'          : wavrange
        }
        save_as_pickle(allLinesSpec, fileName = f"{self.objid}-allLinesSpec.pkls")
        self.plotVerticalStack()
        
        if(len(wavrange)==0):
            print("Provide wavelength ranges and rerun ")
            
        else:
            self.user_wavrange    = wavrange
            self.Nsm              = Nsm
            
            ## Create inverse variance array, from standard deviation
            ## Note that the above could give problems if there are any cases of sigma=0
            invvar = []
            for i in range(len(spectra_interp)):
                invvar.append(sigma[i]**(-2.))
                
            ## TJ: set inverse-variance to 0 to mask out regions outside the wavrange.
            for i in range(len(spectra_interp)):
                invvar[i][ wav[i] < wavrange[i][0] ]  = 0
                invvar[i][ wav[i] > wavrange[i][1] ]  = 0

            ## TJ version of combined average profile :  We have X lines with each line having an ivar. 
            ## Take the inversevariance-weighted mean of all the X lines to get the final combined absorption profile
            f_interp        = np.array(spectra_interp)
            ivar_interp     = np.array(invvar)

            # Inverse-variance weighted mean of the absorption lines
            final_profile   = np.sum(f_interp*ivar_interp, 0) / np.sum(ivar_interp, 0)
            final_ivar      = np.sum(ivar_interp, 0)
            final_error     = 1./np.sqrt(final_ivar)

            # Fitting final profile continuum with polynomial
            if(len(roi)>0):

                mask_continuum0       = wav[0] < roi[0]
                mask_continuum1       = wav[0] > roi[1]
                mask_continuum        = mask_continuum0 | mask_continuum1

                continuum_fit         = np.polyfit(
                    wav[0][mask_continuum],
                    final_profile[mask_continuum],
                    3
                )
                continuum_fit_poly    = np.poly1d(continuum_fit)

                final_profile   = final_profile/continuum_fit_poly(wav[0])
                final_error     = final_error/continuum_fit_poly(wav[0])
                
                continuumFitVals = {
                    'wav'                   : wav[0],
                    'final_profile'         : final_profile,
                    'final_error'           : final_error,
                    'continuum_fit_poly'    : continuum_fit_poly(wav[0])
                }
                save_as_pickle(continuumFitVals, fileName=f"{self.objid}-continuumfit.pkls")
                self.plotContinuumFit()

            #*********************************************#

            self.combined_wavelength    = wav[0]
            self.combined_flux          = final_profile
            self.combined_fluxerr       = final_error
            self.all_lines              = f_interp
            self.all_lines_err          = 1./np.sqrt(ivar_interp)
            self.line_names             = line_name


            computedSpecVals = {
                'source_wav'        : self.combined_wavelength,
                'source_flux'       : self.combined_flux,
                'source_error'      : self.combined_fluxerr,
                'wav'               : wav,
                'spectra_interp'    : f_interp,
                'sigma'             : 1./np.sqrt(ivar_interp),
                'line_name'         : line_name,    
                'wavrange'          : wavrange,           
                'state'             : state
            }
            save_as_pickle(computedSpecVals, fileName = f"{self.objid}-combined.pkls")
            self.plotCombinedLines()
    
    @staticmethod
    def smooth_and_rebin(
        specDict,
        wavmin = -1401,
        wavmax = 1401,
        smooth_sigma = 3,
        rebin_shape = (28,)
    ):
        '''
        Example usage: 
        ## LRIS (R=600) ##
        - First smooth by 8 pixels (200 km/s) --> FWHM ~500 km/s (R=600)
        - Then rebin the array which is currently of shape (112,)---> (14,). 
        - Therefore 14 elements are used to represent the wavelength range -1400, 1400 km/s

        ## R=1000 ##
        - First smooth by 5 pixels (125 km/s) which is FWHM ~300 km/s (R=1000)
        - Then rebin the array which is currently of shape (100,)---> (20). 
        - Therefore 20 elements are used to represent the wavelength range -1250, 1250 km/s


        ## KCWI (R=2000) ##
        - First smooth by 3 pixels (75 km/s) which is FWHM ~ 175 km/s (R=2000)
        - Then rebin the array which is currently of shape (112,)---> (28,). 
        - Therefore 28 elements are used to represent the wavelength range -1400, 1400 km/s

        '''  
        xChosen    = specDict['wav']
        yChosen    = specDict['flux']
        yErrChosen = specDict['error']
        
        ####### Choose a wavelength range ############
        wav_min = wavmin #km/s
        wav_max = wavmax  #km/s
        mask    = (xChosen>wav_min) & (xChosen<wav_max)
            
        ### Smooth the spectra ###    
        flux_smoothed = gaussian_filter1d(
            yChosen[mask],
            sigma=smooth_sigma
        )

        ### Rebin  ###
        flux_smoothed_rebinned = bin_ndarray(
            flux_smoothed, 
            new_shape=rebin_shape, 
            operation='mean'
        )


        wave_rebinned = bin_ndarray(
            xChosen[mask], 
            new_shape=rebin_shape, 
            operation='mean'
        )

        ######## WARNING : This error is only used for chi-square minimization and is not the TRUE error
        sigma_rebinned  = bin_ndarray(
            yErrChosen[mask], 
            new_shape=rebin_shape, 
            operation='mean'
        )

        return({'wav':wave_rebinned, 'flux': flux_smoothed_rebinned, 'error': sigma_rebinned})
      
            
    ##########################
    ### Plotting routines  ###
    ##########################
    
    def plotVerticalStack(self, xlim=[-2000,2000], ylim=[0,2]):
        fooDict = load_pickle(fileName = f"{self.objid}-allLinesSpec.pkls")
        
        spectra_interp    = fooDict['spectra_interp']
        line_name         = fooDict['line_name']
        wav               = fooDict['wav']

        plt.figure(figsize = [7,8])
        
        count = len(self.color_pal)-1

        for i in reversed(range(len(spectra_interp))):
            plt.plot(    
                wav[i],
                spectra_interp[i]+i,
                alpha       = 0.8,
                linewidth   = 3,
                label       = line_name[i],
                color       = self.color_pal[count]
            )
            count = count-1
            
            beautifyPlot({                     
                'xlabel'  : {'xlabel': 'Velocity(km/s)'},
                'ylabel'  : {'ylabel': 'Flux (Normalized)'},

                'xlim'    : {'left': xlim[0],'right': xlim[1]},
                'ylim'    : {'bottom': ylim[0],'top': None},

                
                'tightlayout'   : {},
                'grid'          : {'visible' : True, 'alpha' : 0.6 },
                'legend'        :{
                    'bbox_to_anchor'    : (1, 1),
                    'loc'               : 'upper right',
                    'ncol'              : 1
                }
            })           

        plt.savefig("%s-lines-verticalstack.png"%(self.objid), **ISMgasPlot['savefig'])   
        plt.close()   
        
    def plotContinuumFit(self, xlim=[-2000,2000], ylim=[0,2]):
        fooDict               = load_pickle(fileName=f"{self.objid}-continuumfit.pkls")
        wav                   = fooDict['wav']
        continuum_fit_poly    = fooDict['continuum_fit_poly']
        final_profile         = fooDict['final_profile']
        final_error           = fooDict['final_error']

        plt.figure(figsize = [12,7])
        plt.plot(
            wav, continuum_fit_poly,
            alpha       = 0.8,
            linewidth   = 3,
            linestyle   = '--',
            label       = 'fit-continuum',
            color       = self.color_pal[16]
        )

        plt.plot(
            wav, final_profile,
            alpha       = 0.8,
            linewidth   = 3,
            label       = 'data',
            color       = self.color_pal[16]
        )
        beautifyPlot({  

            'xlabel'  : {'xlabel': 'Velocity(km/s)','fontsize': 20},
            'ylabel'  : {'ylabel': 'Flux', 'fontsize' : 20},

            'xlim'    : {'left': xlim[0],'right': xlim[1]},
            'ylim'    : {'bottom': ylim[0],'top': ylim[1]},

            'xticks':{
                'ticks'       : np.arange(xlim[0],xlim[1],200),
                'fontsize'    : 15,
                'rotation'    : 90
            },
            'yticks':{
                'ticks'       : np.arange(ylim[0],ylim[1],0.5),
                'fontsize'    : 15
            },
            
            'tightlayout'   : {},
            'legend'        : {},
            'grid'          : {'visible' : True, 'alpha' : 0.6 }

        })
        plt.savefig("%s-fit-continuum.png"%(self.objid), **ISMgasPlot['savefig'])
        plt.close()
        
    def plotCombinedLines(self, xlim=[-2000,2000], ylim=[0,2]):
        fooDict = load_pickle(fileName = f"{self.objid}-combined.pkls")
    
        wav             = fooDict['source_wav']
        final_profile   = fooDict['source_flux']
        final_error     = fooDict['source_error']
        state           = fooDict['state']
        
        plt.figure(figsize = [12,7])
        plotWithError(
            x       = wav,
            y       = final_profile,
            yerr    = final_error
        )
        
        beautifyPlot({  

            'xlabel'  : {'xlabel': 'Velocity(km/s)','fontsize': 20},
            'ylabel'  : {'ylabel': 'Flux', 'fontsize' : 20},

            'xlim'    : {'left': xlim[0],'right': xlim[1]},
            'ylim'    : {'bottom': ylim[0],'top': ylim[1]},

            'xticks':{
                'ticks'       : np.arange(xlim[0],xlim[1],200),
                'fontsize'    : 15,
                'rotation'    : 90
            },
            'yticks':{
                'ticks'       : np.arange(ylim[0],ylim[1],0.5),
                'fontsize'    : 15
            },
            
            'tightlayout'   : {},
            'legend'        : {},
            'grid'          : {'visible' : True, 'alpha' : 0.6 }

        })
        
        fooPlotDict = {
            'lowion'    : {'title': "%s - Absorption profile - Low ionization lines "%(self.objid) , 'saveFile': "%s-lowions-velocityprofile.png"%(self.objid)} ,
            'highion'   : {'title': "%s - Absorption profile - High ionization lines "%(self.objid) , 'saveFile': "%s-highions-velocityprofile.png"%(self.objid)},
            'opthin'    : {'title': "%s - Optically thin lines "%(self.objid), 'saveFile': "%s-opthin-velocityprofile.png"%(self.objid)},
            'stellar'   : {'title': "%s - Absorption profile - Stellar lines "%(self.objid), 'saveFile': "%s-stellar-velocityprofile.png"%(self.objid)},
            'fineem'    : {'title': "%s - Fine structure emission lines "%(self.objid), 'saveFile': "%s-fineem-velocityprofile.png"%(self.objid)},
            'nebem'     : {'title': "%s - Nebular emission lines "%(self.objid), 'saveFile': "%s-nebem-velocityprofile.png"%(self.objid)}
        }
        plt.title(fooPlotDict[state]['title'],fontsize = 20)
        plt.savefig(fooPlotDict[state]['saveFile'], **ISMgasPlot['savefig'])
        plt.close()

    def plotSmoothedSpectra(
            self,
            Nsm,
            display               = "python",
            xlim                  = [None,None],
            ylim                  = [None,None],
            printLinesUsed        = False,
            plotDeflectorLines    = False,
            plotStellar           = False 
        ):
        '''
        Plots weighted average spectra after smooted by Nsm sized kernel
        This can be displayed on a simple python matplotlib.


        - Nsm = number of elements to average over
        - display = 'python' or 'js' - displays the spectra using matplotlib or plotly
        - linelists_highz = ISM lines
        - linelists_stellar = Stellar lines
        - linelists_general = rough lines spanning the entire spectral range.

        After plotting, returns the wavlength,flux and sigma arrays used for plotting
        '''
        
        wave, spec, ivar    = self.fetchData()
        wave_rest           = wave/(1+self.zs)

        f_wtd, ww_wtd, sigma_wtd = returnWeightedArray(
            Nsm          = Nsm,
            spec         = spec,
            ivar         = ivar,
            wave_rest    = wave_rest
        )
        
        save_as_pickle({'Nsm':Nsm, 'flux':f_wtd, 'wav': ww_wtd, 'sigma':sigma_wtd}, "%s-smoothedSpectra.pkls"%(self.objid))
    
        if(display=="python"):
            ''' (default) uses matplotlib '''
            plt.figure(figsize=[15,5])
            font_size=15

            linelist_keys   = linelist_highz.keys()
            if(printLinesUsed):
                print(linelist_keys)

            plt.plot(
                ww_wtd,
                f_wtd,
                alpha       = 0.8,
                linewidth   = 2,
                color       = self.color_pal[11]
            )

            if(plotDeflectorLines):
                colorDef = iter(cm.rainbow(np.linspace(0, 1, 5)))
                for k in self.zdef:
                    colorDefChosen = next(colorDef)
                    for i in linelist_keys:

                        if(
                            linelist_highz[i].get("plot") == True and
                            linelist_highz[i].get("emission") == False  and
                            linelist_highz[i].get("lowion") == True or
                            linelist_highz[i].get("highion") == True
                        ):

                            plt.vlines(
                                x           = linelist_highz[i]["lambda"]*(1+k)/(1+self.zs),
                                ymin        = ylim[1]-0.3,
                                ymax        = ylim[1],
                                color       = colorDefChosen,
                                alpha       = 0.8,
                                linestyle   = '-',
                                linewidth   = 2
                            )

                    plt.plot(
                        0,0,
                        linewidth   = 1,
                        color       = colorDefChosen,
                        alpha       = 0.4,
                        label       = "Inter-abs (z=%.2f)"%(k)
                    )


            if(printLinesUsed):
                print(linelist_keys)

            for i in linelist_keys:
                if(linelist_highz[i].get("lowion") == True and linelist_highz[i].get("plot") == True):
                    plt.vlines(
                        x           = linelist_highz[i]["lambda"],
                        ymin        = ylim[0],
                        ymax        = ylim[1],
                        color       = 'blue',
                        alpha       = 0.8,
                        linestyle   = '--',
                        linewidth   = 2
                    )

                if(linelist_highz[i].get("highion")==True and linelist_highz[i].get("plot")==True):
                    plt.vlines(
                        x           = linelist_highz[i]["lambda"],
                        ymin        = ylim[0],
                        ymax        = ylim[0]+0.3,
                        color       = 'purple',
                        alpha       = 0.8,
                        linestyle   = '--',
                        linewidth   = 2
                    )

                if(plotStellar and linelist_highz[i].get("stellar")==True and linelist_highz[i].get("plot")==True):

                    plt.vlines(
                        x           = linelist_highz[i]["lambda"],
                        ymin        = ylim[0],
                        ymax        = ylim[0]+0.3,
                        color       = 'coral',
                        alpha       = 0.8,
                        linestyle   = '--',
                        linewidth   = 2
                    )

            plt.plot(
                0, 0,
                linewidth   = 5,
                color       = 'blue',
                alpha       = 0.8,
                label       = "ISM-lowion"
            )
            plt.plot(
                0, 0,
                linewidth   = 5,
                color       = 'purple',
                alpha       = 0.8,
                label       = "ISM-highion"
            )

            if(plotStellar):
                plt.plot(
                    0,0,
                    linewidth   = 5,
                    color       = 'coral',
                    alpha       = 0.8,
                    label       = "Stellar"
                )
                
            beautifyPlot({  
                'title'   : {'label': '%s (z=%.5f)'%(self.objid,self.zs), 'fontsize' : font_size},

                'xlabel'  : {'xlabel': "Rest frame wavelength (in $\AA$)",'fontsize': font_size},
                'ylabel'  : {'ylabel': "Normalized Flux", 'fontsize' : font_size},

                'xlim'    : {'left': xlim[0],'right': xlim[1]},
                'ylim'    : {'bottom': ylim[0],'top': ylim[1]},

                'xticks':{
                    'ticks'       : np.arange(xlim[0],xlim[1],100),
                    'fontsize'    : 15,
                },
                'yticks':{
                    'ticks'       : np.arange(ylim[0],ylim[1],0.5),
                    'fontsize'    : 15
                },
                
                'tightlayout'   : {},
                'legend'        : {'loc': 'upper right', 'fontsize': font_size},
                'grid'          : {'visible' : True, 'alpha' : 0.6, 'linestyle':'--', 'linewidth': 1}

            })

            plt.savefig("%s-smooth-spectra.png"%(self.objid), **ISMgasPlot['savefig'])
            plt.close()