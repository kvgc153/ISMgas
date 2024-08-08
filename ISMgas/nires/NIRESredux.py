##############################################
### Handles NIRES data reduction using nsx ###
### Needs better docs and better formatting ##
##############################################
import matplotlib.pyplot as plt
import numpy as np
import os 
from astropy.io import fits
from astropy.table import Table
from astropy.visualization import (ZScaleInterval, ImageNormalize,SqrtStretch, SquaredStretch)
from matplotlib.widgets import Slider, Button, RadioButtons
import mplcursors

## Custom packages
from ISMgas.linelist import linelist_NIRES
from ISMgas.globalVars import *


#################
## Inititalize ##
## sp3 - top most order in detector ##
## slit : Gives the min and max coordinates of the slit in a given order. ##
#################

wav_sol_folder    = '../calib_files/'
wav_sol_files     = [
                 'skylines-sp3.csv',
                 'skylines-sp4.csv',
                 'skylines-sp5.csv',
                 'skylines-sp6.csv',
                 'skylines-sp7.csv'
                 ]
wav_minmax    = [
              [24693.851,18900.119],
              [18550.272,14203.881],
              [14860.462,11380.506],
              [12401.956,9495.88],
              [10646.087,9408.54] 
             ]

slitLengthpxs     = 112
offset            = slitLengthpxs + 86

NIRES_calib_configs = {
    'sp3': {
        'slit'            : [802,802 + slitLengthpxs],
        'wav_solution'    : 'skylines-sp3.csv',
        'wav_minmax'      : [24693.851,18900.119],
        'offsetVals'      : [0,1,2,3,4]
    },
    'sp4': {
        'slit'            : [602,602 + slitLengthpxs],
        'wav_solution'    : 'skylines-sp4.csv',
        'wav_minmax'      : [18550.272,14203.881],
        'offsetVals'      : [-1,0,1,2,3]
        
    },
    'sp5': {
        'slit'            : [402,402 + slitLengthpxs],
        'wav_solution'    : 'skylines-sp5.csv',
        'wav_minmax'      : [14860.462,11380.506],
        'offsetVals'      : [-2,-1,0,1,2]
        
    },
    'sp6': {
        'slit'            : [202,202 + slitLengthpxs],
        'wav_solution'    : 'skylines-sp6.csv',
        'wav_minmax'      : [12401.956,9495.88]  ,
        'offsetVals'      : [-3,-2,-1,0,1]
        
    },
    'sp7': {
        'slit'            : [2, 2 + slitLengthpxs] ,
        'wav_solution'    : 'skylines-sp7.csv',
        'wav_minmax'      : [10646.087,9408.54] ,
        'offsetVals'      : [-4,-3,-2,-1,0]
    },
}

####################
## Read wave maps ##
####################
wavelength_scale              = fits.getdata(NIRES_wavemap_file)
wavelength_scale              = wavelength_scale.astype(int)

wavelength_scale_corrected    = fits.getdata(NIRES_wavemap_file_corrected)
wavelength_scale_corrected    = wavelength_scale_corrected.astype(int)


class NIRESredux:
    def __init__(self,**kwargs):
        self.baseFolder     = kwargs.get('baseFolder')
        self.AFrames        = kwargs.get('A')
        self.AFrames        = [self.baseFolder+i for i in self.AFrames]

        self.BFrames        = kwargs.get('B')
        self.BFrames        = [self.baseFolder+i for i in self.BFrames]
       
        self.slitAFrames    = kwargs.get('slitA')
        self.slitAFrames    = [self.baseFolder+i for i in self.slitAFrames]

        self.slitBFrames    = kwargs.get('slitB')
        self.slitBFrames    = [self.baseFolder+i for i in self.slitBFrames]

        self.objid          = kwargs.get('objid')
        self.reducedABFile  = ''

        self.slitPosition2D()
        self.redux2D()
        
    def slitPosition2D(self):
        '''
        Make a 2D image of the slit position
        '''
        A_data    = [[fits.getdata(i)] for i in self.slitAFrames ]
        B_data    = [[fits.getdata(i)] for i in self.slitBFrames ]

        ## Take mean of frames 
        A_data        = np.array(A_data)
        A_data_mean   = np.mean(A_data,axis=0)
        B_data        = np.array(B_data)
        B_data_mean   = np.mean(B_data,axis=0)

        ## Store A-B of the mean files 
        fits.writeto(self.objid+"_slitPosition.fits",A_data_mean-B_data_mean,overwrite=True)
        os.system("ds9 %s"%(self.objid+"_slitPosition.fits &"))

        
    def redux2D(self):
        '''
        Run nsx in autox mode
        '''
        A_data    = [[fits.getdata(i)] for i in self.AFrames ]
        B_data    = [[fits.getdata(i)] for i in self.BFrames ]
        
        ## Take mean of frames 
        A_data        = np.array(A_data)
        A_data_mean   = np.mean(A_data,axis=0)
        fits.writeto(self.objid+"_A.fits",A_data_mean,overwrite=True)

        B_data        = np.array(B_data)
        B_data_mean   = np.mean(B_data,axis=0)
        fits.writeto(self.objid+"_B.fits",B_data_mean,overwrite=True)
        
        ## Reduce the coadded frames 
        file1   = self.objid+"_A.fits"
        file2   = self.objid+"_B.fits"
        
        ## Store A-B of the mean files 
        fits.writeto(self.objid+"A_B.fits",A_data_mean-B_data_mean,overwrite=True)

        ## Shell commands to reduce the data
        command   = NIRES_nsxPath + file1 + ' ' +  file2 + ' -autox'
        os.system(command)
        # command   = 'rm *.tbl'
        # os.system(command)
        # command   = 'rm *.log'
        # os.system(command)
        command   = 'rm *.draw'
        os.system(command)
        
        self.reducedABFile = self.objid+"_A-"+self.objid[-2:]+"_B-corrected.fits"
        print(self.reducedABFile)
        
        ## Extract each order and store them 
        dataReduced = fits.getdata(self.reducedABFile)
        
        ## bad code ---
        for i in  NIRES_calib_configs.keys():
            slitMin, slitMax    = NIRES_calib_configs[i]['slit']
            dataSliced          = dataReduced[slitMin: slitMax, :]
            fits.writeto(i+".fits", dataSliced, overwrite=True)
                
    def plotReducedSpectra(self,guessZ,imshowParams={'vmin':-7, 'vmax':3.5}):
        ## Reduced spectra
        data    = fits.getdata(self.reducedABFile)
        norm    = ImageNormalize(data, interval=ZScaleInterval())

        plt.figure(figsize=(10,5))
        plt.imshow(data,origin='lower',cmap='gray',**imshowParams)
        self.plot_emission_lines(guessZ)
        os.system("ds9 %s %s -zoom to fit -lock frame image -lock scale -scale zscale -region %s &"%(self.reducedABFile, self.objid+"A_B.fits",self.objid+"_lines.reg"))
              

    def plot_emission_lines(self,z):
        ## Superimpose the lines on top of the spectra
        f = open(self.objid+"_lines.reg",'w+')
        f.write('''# Region file format: DS9 version 4.1
global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
physical
        ''')
        for i in linelist_NIRES: 
            element               = i[0]
            wav                   = i[1]*(1+z)
            x,y                   = np.where(wavelength_scale == int(wav))
           
            try:
                plt.scatter(y[0],x[0], marker = 'o',s=20,color = 'pink')
                f.write("circle %d %d 40 # color=red text={%s} \n"%(y[0],x[0],element))
                print(y[0],x[0],element,wav)
                
                plt.text(y[0],x[0]+50,element,fontsize = 20,color = 'pink')
            except:
                pass
            
        f.close()    

    def redux1D(
            self, 
            order = 'sp4', 
            delta = 40, 
            Amin = 630, 
            Amax = 655, 
            z = 2, 
            specPlotParams = {
                'figsize' : (12,5),
                'xlim'    : 'default',
                'ylim'    : 'default',
            }
        ):
        
        ## Plotting the region to be  extracted --  160 pixels is roughly 18''
        Bmin          = Amin + delta
        Bmax          = Amax + delta
        offset_vals   = NIRES_calib_configs[order]['offsetVals']
        
        data          = fits.getdata(self.reducedABFile)

        plt.figure(figsize=(10,7))
        for i in offset_vals:    
            plt.imshow(data,origin='lower',vmin=-10,vmax=10,cmap ='gray')
            plt.axhspan(Amin-i*offset,Amax-i*offset,color='purple',alpha=0.4)
            plt.axhspan(Bmin-i*offset,Bmax-i*offset,color='pink',alpha=0.4)
        plt.tight_layout()
        plt.savefig(f"{self.objid}_orders.png",dpi=300)
        plt.close()
        
        ## Make 2D masks ##
        for count,fooKeys in enumerate(NIRES_calib_configs.keys()):
            t   = Table.read(wav_sol_folder+ wav_sol_files[count],format='ascii.csv')
            wavelength    = t['wavelength']
            col           = t['col']          
            
            if(count < 4): ## Need to rewrite this to account for the last index error
                ## xlim 
                if(specPlotParams['xlim']  == 'default'):
                    mask1 = wavelength > wav_minmax[count][1]
                    mask2 = wavelength < wav_minmax[count][0]
                    mask  = mask1 | mask2
                    
                    fooData = fits.getdata(fooKeys + ".fits")
                    fooDataShape = np.shape(fooData)
                    newData = []
                    for i in range(fooDataShape[0]):
                        newData.append(fooData[i,:][mask])
                    fits.writeto(fooKeys + "-masked.fits", np.fliplr((np.array(newData))), overwrite=True)
                    
                    fooData1    = fits.getdata(fooKeys + "-masked.fits")
                    foorNorm1   = ImageNormalize(fooData1, interval=ZScaleInterval())
                    plt.imshow(fooData1,origin='lower',cmap='gray', norm=foorNorm1)
                    plt.axis('off')
                    plt.tight_layout()
                    plt.savefig(fooKeys + "-masked.png", dpi = 150)
                    
                    
                else: 
                    mask1 = wavelength > specPlotParams['xlim'][count][0]*10000
                    mask2 = wavelength < specPlotParams['xlim'][count][1]*10000
                    mask  = mask1 & mask2
                    
                    fooData = fits.getdata(fooKeys + ".fits")
                    fooDataShape = np.shape(fooData)
                    newData = []
                    for i in range(fooDataShape[0]):
                        newData.append(fooData[i,:][mask])
                    fits.writeto(fooKeys + "-masked.fits", np.fliplr((np.array(newData))), overwrite=True)
 
 
                    fooData1    = fits.getdata(fooKeys + "-masked.fits")
                    foorNorm1   = ImageNormalize(fooData1, interval=ZScaleInterval())
                    plt.imshow(fooData1,origin='lower',cmap='gray', norm=foorNorm1)
                    plt.axis('off')
                    plt.tight_layout()
                    plt.savefig(fooKeys + "-masked.png", dpi = 150)                   
                        
        count = 1 
        for i in offset_vals:
            plt.figure(figsize = specPlotParams['figsize'])

            data_f1       = sum(data[Amin-i*offset:Amax-i*offset,:]) - sum(data[Bmin-i*offset:Bmax-i*offset,:])
            
            t             = Table.read(wav_sol_folder+ wav_sol_files[count-1],format='ascii.csv')
            wavelength    = t['wavelength']
            col           = t['col']
            sky_flux      = t['sky']
            
            
            if(count==5):
                plt.plot(
                    wavelength/10000., 
                    data_f1[:1024],
                    color   = 'black',
                    label   = 'spectra'
                    )
                
            else :
                plt.plot(
                    wavelength/10000., 
                    data_f1,
                    color   = 'black',
                    label   = 'spectra'
                    )
                
            plt.plot(
                    wavelength/10000.,
                    sky_flux*10-100,
                    linewidth   = 2,
                    color       = 'darkorange',
                    label       = 'scaled sky'
            )
            
            skymask             = np.array(sky_flux) > 0.5
            wavelength_masked   = wavelength[skymask]
            skyflux_masked      = sky_flux[skymask]
            
            for foox,fooy in zip(wavelength_masked, skyflux_masked):
                plt.axvline([foox/10000.], color='gray', linewidth= 5, alpha =0.04*(fooy))
            

            trans = plt.gca().get_xaxis_transform()             
            for i in linelist_NIRES:
                plt.axvline([i[1]*(1+z)/10000.],alpha=0.6)

                plt.gca().annotate(
                        i[0],
                        xy          = (i[1]*(1+z)/10000., 1.05),
                        xycoords    = trans,
                        fontsize    = 20,
                        rotation    = 0,
                        color       = 'b'
                )     

            ## xlim 
            if(specPlotParams['xlim']  == 'default'):
                plt.xlim([wav_minmax[count-1][1]/10000.,wav_minmax[count-1][0]/10000.])
            else: 
                plt.xlim(specPlotParams['xlim'][count-1])

            ## ylim 
            if(specPlotParams['ylim']  == 'default'):
                plt.ylim([-200,None])
            else: 
                plt.ylim(specPlotParams['ylim'][count-1])

            plt.xlabel("Observed Wavelength (in microns)", fontsize = 18)
            plt.ylabel("Flux (arbitrary units)", fontsize = 18 )
            plt.xticks(fontsize= 15)
            plt.yticks(fontsize= 15)
            plt.legend(fontsize = 15)
            plt.tight_layout()

            plt.savefig(f"{self.objid}-{count-1}.png", dpi=100)
            count+=1
        


    def plotInteractiveView(self):
        '''
        This is mostly to check the wavemap values. Could be used for redshift estimation but not recommended
        '''

        data    = fits.getdata(self.objid+"A_B.fits")[0,:,:]

        fig     = plt.figure(figsize=(10,5))
        ax      = fig.add_subplot(111)
        fig.subplots_adjust(left = 0.25, bottom=0.25)
        min0    = 0
        max0    = 25000
        im1     = ax.imshow(data,vmin=-10,vmax=10.94,cmap='gray',origin='lower')
        wv      = ax.imshow(fits.getdata(wavemap_file),origin='lower',alpha= 0)

        axmin   = fig.add_axes([0.25, 0.1, 0.65, 0.03])
        axmax   = fig.add_axes([0.25, 0.15, 0.65, 0.03])

        smin    = Slider(axmin, 'Min', -20, 20, valinit=-10)
        smax    = Slider(axmax, 'Max', -20, 20, valinit=10.94)

        def update(val)             : 
            im1.set_clim([smin.val,smax.val])
            fig.canvas.draw()
        smin.on_changed(update)
        smax.on_changed(update)


        mplcursors.cursor(wv,multiple=True)
        plt.show()


        
        
