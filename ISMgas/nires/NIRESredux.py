##############################################
### Handles NIRES data reduction using nsx ###
### Needs better docs and better formatting ##
##############################################
import matplotlib.pyplot as plt
import numpy as np
import os 
from astropy.io import fits,ascii
from astropy.table import Table
from astropy.visualization import (ZScaleInterval, ImageNormalize,SqrtStretch, SquaredStretch)
from matplotlib.widgets import Slider, Button, RadioButtons
# import mplcursors

from scipy.signal import correlate
from scipy.optimize import curve_fit

## Custom packages
from ISMgas.linelist import linelist_NIRES, linelist_SDSS, linelist_NIRES_detailed,linelist_SDSS_emission
from ISMgas.globalVars import *
from ISMgas.SupportingFunctions import interpolateData, save_as_pickle, load_pickle,wavelengthToVelocity
from ISMgas.visualization.fits import ScaleImage
import glob 
import shutil

#################
## Inititalize ##
## sp3 - top most order in detector ##
## slit : Gives the min and max coordinates of the slit in a given order. ##
#################

wav_sol_folder    = 'calib_files/'
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



def initNIRES(filenames):
    """Returns a dict with files to reduce.

    Args:
        filenames (_type_): _description_

    Returns:
        _type_: _description_
    """
    filesToReduce = {}
    for i in filenames:
        try:
            header = fits.getheader(i)
            # print(i.split('/')[-1].split('_')[-1],header['OBJECT']

            dataExisting = filesToReduce.get(header['OBJECT'])
            if(dataExisting is None):
                filesToReduce[header['OBJECT']] = [i.split('/')[-1]]
                
            else:
                dataExisting.append(i.split('/')[-1])
                filesToReduce[header['OBJECT']] = dataExisting
                
        
        except:
            pass
        
    return(filesToReduce)



class NIRESredux:
    def __init__(self,**kwargs):
        self.baseFolder     = kwargs.get('baseFolder',"") ## Assume current folder 
        self.AFrames        = kwargs.get('A')
        self.AFrames        = [self.baseFolder+i for i in self.AFrames]

        self.BFrames        = kwargs.get('B')
        self.BFrames        = [self.baseFolder+i for i in self.BFrames]
       
        self.slitAFrames    = kwargs.get('slitA',[])
        self.slitAFrames    = [self.baseFolder+i for i in self.slitAFrames]

        self.slitBFrames    = kwargs.get('slitB',[])
        self.slitBFrames    = [self.baseFolder+i for i in self.slitBFrames]

        self.objid          = kwargs.get('objid')
        self.reducedABFile  = ''
        
        self.wavOffsetResults = {} 
        self.wavOffsetArray = []
        self.wavOffset =  0 
        self.wavOffsetStd = 0
        
        
        self.dataProduct    = {}
        


        if(len(self.slitAFrames)>0):
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

        ## Show user the slit position
        
        plt.figure(figsize=(10,10))
        dataSlit = A_data_mean[0]-B_data_mean[0]
        
        plt.imshow(dataSlit,origin='lower',cmap='gray', vmin = np.percentile(dataSlit,10), vmax = np.percentile(dataSlit,90))
        plt.tight_layout()
        plt.savefig(self.objid+"_slitPosition.png",dpi=150)

        ## Store A-B of the mean files 
        hdu = fits.PrimaryHDU()
        hdu.data = A_data_mean-B_data_mean
        hdu.header = fits.getheader(self.slitAFrames[0])
        hdu.writeto(self.objid+"_slitPosition.fits",overwrite=True)
        os.system("ds9 %s"%(self.objid+"_slitPosition.fits &"))

        
    def redux2D(self, mode='autox',bk='',sp=''):
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
    

        if(mode == 'autox'):
            ## Shell commands to reduce the data
            command   = NIRES_nsxPath + file1 + ' ' +  file2 + ' -autox'
            os.system(command)

        self.data1Corrected = fits.getdata(self.objid+"_A-corrected.fits")
        self.data2Corrected = fits.getdata(self.objid+"_B-corrected.fits")
        
        ## Store A-B of the mean files 
        fits.writeto(self.objid+"A_B.fits",A_data_mean-B_data_mean,overwrite=True)


        self.reducedABFile = self.objid+"_A-"+self.objid[-2:]+"_B-corrected.fits"
        print(self.reducedABFile)

    
    def findWavelengthOffset(
            self, 
            order = 'sp4', 
            delta = 40, 
            Amin = 630, 
            Amax = 675, 
            offsetWav = 0,
            scalesky = 10,
            figsize= (20,7),
            ylim = [None,None]
        ):
        
        offset_vals   = NIRES_calib_configs[order]['offsetVals']
        
        data          = self.data1Corrected

        ## Show user which range is being extracted ##
        plt.figure(figsize=(10,7))
        for i in offset_vals:    
            plt.imshow(data,origin='lower',vmin=-10,vmax=10,cmap ='gray')
            plt.axhspan(Amin-i*offset,Amax-i*offset,color='purple',alpha=0.4)
        plt.tight_layout()
        ##############################################
        
        wavelengthOffset = [] ## Array that stores the wavelength offset
        sky_observed    = [] ## Array that stores the observed sky spectra
        
        count = 1 
        plt.figure(figsize=(15,5))
        for i in offset_vals:          
            
            if(count<5): # We will only do this for the first 4 orders 
                
                data_f1       = np.sum(data[Amin-i*offset:Amax-i*offset,:], axis=0)
                sky_observed.append(data_f1)
                
                
                t             = Table.read(wav_sol_folder+ wav_sol_files[count-1],format='ascii.csv')
                wavelength    = t['wavelength']/10000.
                col           = t['col']
                sky_flux      = t['sky']
                
                
                ## Interpolate the data to the same wavelength scale of 0.1 Angstroms
                deltaWav_new = 0.00001
                wavelength_new = np.arange(wavelength[-1],wavelength[0],deltaWav_new)
                
                data_f1 = interpolateData(wavelength, data_f1, wavelength_new)
                sky_flux = interpolateData(wavelength, sky_flux, wavelength_new)
                
                
                ## Cross correlate the spectra with the template
                spectra = data_f1
                template = sky_flux*scalesky
                
                # Normalize the data
                spectra = (spectra - np.min(spectra)) / (np.max(spectra) - np.min(spectra))
                template = (template - np.min(template)) / (np.max(template) - np.min(template))
                
    
                correlation = correlate(spectra, template, mode='full')
                shift_index = np.argmax(correlation) - (len(template) - 1)
                
                plt.subplot(1,4,count)
                plt.plot(correlation)
                
                print("Shift index: ", shift_index)
                print("Shift in wavelength: ", shift_index*deltaWav_new)
                
                wavelengthOffset.append(shift_index*deltaWav_new)
                  
            count+=1
            
        plt.suptitle("Cross correlation of the spectra with the sky template")
        plt.tight_layout()
        print("Wavelength offset: ", np.median(wavelengthOffset), '+-', np.std(wavelengthOffset))
        
        self.wavOffset = np.median(wavelengthOffset)
        self.wavOffsetStd = np.std(wavelengthOffset)
        self.wavOffsetArray = wavelengthOffset
        self.wavOffsetResults = {
            'wavOffset': self.wavOffset,
            'wavOffsetStd': self.wavOffsetStd,
            'offset_vals': {
                'sp4': wavelengthOffset[3],
                'sp5': wavelengthOffset[2],
                'sp6': wavelengthOffset[1],
                'sp7': wavelengthOffset[0]
            },
            'sky_observed': {
                'sp4': sky_observed[3],
                'sp5': sky_observed[2],
                'sp6': sky_observed[1],
                'sp7': sky_observed[0]
            }
        }
        
        ## Diagonstic plot
        count = 1 

        for i in offset_vals:     
            if(count<5):     
                data_f1       = np.sum(data[Amin-i*offset:Amax-i*offset,:], axis=0)
                
                t             = Table.read(wav_sol_folder+ wav_sol_files[count-1],format='ascii.csv')
                wavelength    = t['wavelength']/10000. 
                col           = t['col']
                sky_flux      = t['sky']
                
                
                plt.figure(figsize=figsize)
                plt.plot(wavelength - self.wavOffsetArray[count-1], data_f1, color='black') 
                plt.plot(wavelength, sky_flux*scalesky , color='orange') ## This is the 'true sky'
                plt.xlabel("Wavelength (in microns)", fontsize = 18)
                plt.ylabel("Flux (arbitrary units)", fontsize = 18)
                plt.title(f"Order {count+2}", fontsize = 18)
                plt.ylim(ylim)
                count+=1
                plt.tight_layout()
                plt.savefig(f"{self.objid}_sp{count+2}_wavsolution.png",dpi=100)
    
    
    def plotReducedSpectra(self,guessZ, detailed = False):
        self.plot_emission_lines(guessZ, detailed= detailed)
        os.system("ds9 %s %s -zoom to fit -lock frame image -lock scale -scale zscale -region %s &"%(self.reducedABFile, self.objid+"A_B.fits",self.objid+"_lines.reg"))
              

    def plot_emission_lines(self,z,detailed = False):
        ## Superimpose the lines on top of the spectra
        f = open(self.objid+"_lines.reg",'w+')
        f.write('''# Region file format: DS9 version 4.1
global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
physical
        ''')
        if(detailed):
            for i in linelist_NIRES_detailed: 
                element               = i[0]
                wav                   = i[1]*(1+z)
                x,y                   = np.where(wavelength_scale == int(wav))
            
                try:
                    # plt.scatter(y[0],x[0], marker = 'o',s=20,color = 'pink')
                    f.write("circle %d %d 40 # color=red text={%s} \n"%(y[0],x[0],element))
                    print(y[0],x[0],element,wav)
                    
                    # plt.text(y[0],x[0]+50,element,fontsize = 20,color = 'pink')
                except:
                    pass
            
        else:
            for i in linelist_NIRES: 
                element               = i[0]
                wav                   = i[1]*(1+z)
                x,y                   = np.where(wavelength_scale == int(wav))
            
                try:
                    # plt.scatter(y[0],x[0], marker = 'o',s=20,color = 'pink')
                    f.write("circle %d %d 40 # color=red text={%s} \n"%(y[0],x[0],element))
                    print(y[0],x[0],element,wav)
                    
                    # plt.text(y[0],x[0]+50,element,fontsize = 20,color = 'pink')
                except:
                    pass
            
        f.close()    

    def redux1D(
            self, 
            order = 'sp4', 
            delta = 40, 
            Amin = 630, 
            Amax = 655, 
            scaleLimits = [20,85],
            waveLimits = [1,99.99],
            z = 2,
            scaleAlpha = 1
        ):
        
        Bmin          = Amin + delta
        Bmax          = Amax + delta
        slit_vals     = NIRES_calib_configs[order]['slit']
        
        print("Offsets from base of slit for A : ", Amin - slit_vals[0], Amax - slit_vals[0])
        print("Offsets from base of slit for B : ", Bmin - slit_vals[0], Bmax - slit_vals[0])

        offset_vals   = NIRES_calib_configs[order]['offsetVals']
        
        data          = fits.getdata(self.reducedABFile)
        
        
        ## Show user which range is being extracted ##
        plt.figure(figsize=(10,7),dpi=200)
        for i in offset_vals:    
            plt.imshow(data,origin='lower',vmin=-10,vmax=10,cmap ='gray')
            
            plt.axhspan(Amin-i*offset,Amax-i*offset,color='purple',alpha=0.8*scaleAlpha,fill=False)
            plt.axhspan(Bmin-i*offset,Bmax-i*offset,color='pink',alpha=0.8*scaleAlpha, fill = False,)
            
        plt.tight_layout()
        plt.show()
        ##############################################
        
        ## Make final dataproduct 
        dataProduct    = {
            'z' : z,
            'objid': self.objid,
            'A': self.AFrames,
            'B': self.BFrames,
            'userInputs':{  
                'Amin': Amin,
                'Amax': Amax,
                'Bmin': Bmin,
                'Bmax': Bmax,
                'delta': delta,
                'order': order
            } , 
            'slitA': self.slitAFrames,
            'slitB': self.slitBFrames,
            'reducedABFile': self.reducedABFile,
            'wavOffsetResults': self.wavOffsetResults,
            'wavOffsetArray': self.wavOffsetArray,
            'wavOffset': self.wavOffset,
            'wavOffsetStd': self.wavOffsetStd,
            'comments': '',
            'sp3':{
                'lambda' : [],
                'flux'   : [],
                'sky_model_lambda' : [], 
                'sky_model'    : []
            },
            'sp4':{
                'lambda' : [],
                'flux'   : [],
                'sky_model_lambda' : [], 
                'sky_model'    : []
            },
            'sp5':{
                'lambda' : [],
                'flux'   : [],
                'sky_model_lambda' : [], 
                'sky_model'    : []
            },
            'sp6':{
                'lambda' : [],
                'flux'   : [],
                'sky_model_lambda' : [], 
                'sky_model'    : []
            },
            'sp7':{
                'lambda' : [],
                'flux'   : [],
                'sky_model_lambda' : [], 
                'sky_model'    : []
            },
        }      
        
        
        ##############################################
        count = 1 
        for i in offset_vals:
            plt.figure(figsize = (20,8), dpi=100)
            
            plt.subplot(2,1,1)

            ## A extraction 
            data_Aframe   = fits.getdata(self.objid+"_A-corrected.fits")
            data_Bframe   = fits.getdata(self.objid+"_B-corrected.fits")
            
            data_Aframe_2d = data_Aframe[Amin-i*offset:Amax-i*offset,:]- data_Aframe[Bmin-i*offset:Bmax-i*offset,:]
            data_Bframe_2d = data_Bframe[Bmin-i*offset:Bmax-i*offset,:]- data_Bframe[Amin-i*offset:Amax-i*offset,:]
            
            data_Aframe_sum = np.sum(data_Aframe_2d,axis=0)
            data_Bframe_sum = np.sum(data_Bframe_2d,axis=0)
            
            data_f1 = data_Aframe_sum + data_Bframe_sum
            
            
            t             = Table.read(wav_sol_folder+ wav_sol_files[count-1],format='ascii.csv')
            
            wavelength             = t['wavelength']/10000. ## microns
            if(count==5): ## Order 7 has 1024 pixels and is smaller
                wavelengthoffset       = 0 # Note this in the docs that we dont have an offset for this order 
            else:    
                wavelengthoffset       = self.wavOffsetArray[count-1]
            wavelengthCorrected    = wavelength  - wavelengthoffset ## Correct for the offset
            
            col           = t['col']
            sky_flux      = t['sky']
                        
            if(count==5): ## Order 7 has 1024 pixels and is smaller
                dataProduct['sp'+ str(count+2)]['lambda'] = wavelengthCorrected.data
                dataProduct['sp'+ str(count+2)]['flux']   = data_f1[:1024]
                
                
                dataProduct['sp'+ str(count+2)]['sky_model_lambda'] = wavelength.data
                dataProduct['sp'+ str(count+2)]['sky_model']        = sky_flux.data

                
                plt.plot(
                    wavelengthCorrected, 
                    data_f1[:1024],
                    color   = 'black',
                    label   = 'spectra'
                    )
                
            else :
                dataProduct['sp'+ str(count+2)]['lambda'] = wavelengthCorrected.data
                dataProduct['sp'+ str(count+2)]['flux']   = data_f1
                
                
                dataProduct['sp'+ str(count+2)]['sky_model_lambda'] = wavelength.data
                dataProduct['sp'+ str(count+2)]['sky_model']        = sky_flux.data

                plt.plot(
                    wavelengthCorrected, 
                    data_f1,
                    color   = 'black',
                    label   = 'spectra'
                    )
                
            plt.plot(
                    wavelength, ## Use the original uncorrected wavelength for the sky as its the true sky
                    sky_flux*10-100,
                    linewidth   = 2,
                    color       = 'darkorange',
                    label       = 'scaled sky'
            )
            

            trans = plt.gca().get_xaxis_transform()             
            for i in linelist_SDSS_emission.keys():
                line = linelist_SDSS_emission[i]['lambda']
                
                ## Plot only if the line is within the wavelength range
                if(line*(1+z)/10000. > wav_minmax[count-1][1]/10000. and line*(1+z)/10000. < wav_minmax[count-1][0]/10000.):
                
                    plt.axvline([line*(1+z)/10000.],alpha=0.6)

                    plt.gca().annotate(
                            linelist_SDSS_emission[i]['pltLabel'],
                            xy          = (line*(1+z)/10000., 1.05),
                            xycoords    = trans,
                            fontsize    = 10,
                            rotation    = 90,
                            color       = 'b'
                    )     



            plt.xlabel("Observed Wavelength (in microns)", fontsize = 18)
            plt.ylabel("Flux (arbitrary units)", fontsize = 18 )
            plt.title(f"Objid:{self.objid} z={z} Order {count+2}\n\n", fontsize = 18)
            plt.xticks(fontsize= 15)
            plt.yticks(fontsize= 15)
            plt.legend(fontsize = 15)
            plt.xlim([wavelength[-1],wavelength[0]])
            plt.ylim([-150,np.percentile(data_f1,waveLimits[1])])
            plt.tight_layout()

          
            
            
            ### Lets also show the relevant NIRES data
        
            plt.subplot(2,1,2)
            slitCoords   = NIRES_calib_configs['sp'+str(count+2)]['slit']
            ## Extract the data from the slit
            
            ## Make a cutout of the slit to find the vmin and vmax values 
            dataSlit     = data[slitCoords[0]:slitCoords[1],:]
            vmin = np.percentile(dataSlit,scaleLimits[0])
            vmax = np.percentile(dataSlit,scaleLimits[1])
            
            if(count<5):
                ## Make wavelegth 2D image and save it as a fits file
                hdu = fits.PrimaryHDU()
                wav2D  = np.zeros(dataSlit.shape)
                wav2D[:,:] = wavelengthCorrected
                hdu.data = [np.fliplr(dataSlit), np.fliplr(wav2D)]
                hdu.writeto(f"{self.objid}-sp{count+2}_dataCube.fits",overwrite=True)            
            
            if(count==5): ## Order 7 has 1024 pixels and is smaller
                dataSlit = data[:,:1024]
                dataSlit = np.fliplr(dataSlit)
                plt.imshow(
                    dataSlit,
                    origin='lower',
                    cmap='gray',
                    vmin = vmin,
                    vmax = vmax,
                )
                for i in offset_vals:                        
                    plt.axhspan(Amin-i*offset,Amax-i*offset,color='cyan',alpha=scaleAlpha,fill=False)
                    plt.axhspan(Bmin-i*offset,Bmax-i*offset,color='cyan',alpha=scaleAlpha, fill = False)
                
            else: 
                dataSlit = np.fliplr(data)
                plt.imshow(
                    dataSlit,
                    origin='lower',
                    cmap='gray',
                    vmin = vmin,
                    vmax = vmax,
                )
                for i in offset_vals:                        
                    plt.axhspan(Amin-i*offset,Amax-i*offset,color='cyan',alpha=scaleAlpha,fill=False)
                    plt.axhspan(Bmin-i*offset,Bmax-i*offset,color='cyan',alpha=scaleAlpha, fill = False)
                    
            plt.ylim([slitCoords[0],slitCoords[1]])
            plt.axis('off')

            plt.tight_layout()
            count+=1
            
            
            plt.savefig(f"{self.objid}-{count-1}.png", dpi=100)
            
        self.dataProduct = dataProduct
                
            
    def make2DZoomIn(self):
        """
        Plots a zoom in of the 2D image for the lines
        
        """
        z = self.dataProduct['z'] ## Get redshift from last step
        plt.figure(dpi=200, figsize=(10,20))
        cnt = 1
        for count in range(4):
            data = fits.getdata(f"{self.objid}-sp{count+3}_dataCube.fits")
            wave = data[1][0]/(1+z) ## Get the wavelength 
      
            
            for line in linelist_SDSS.keys():
                ## Check if line is in the wavelength of the order 
                checkWave = (linelist_SDSS[line]['lambda']/1e4 >wave[0]) & (linelist_SDSS[line]['lambda']/1e4 <wave[-1])
                if(checkWave):
                    vmin = np.percentile(data[0],5)
                    vmax = np.percentile(data[0],95)
                    
                    
                    # Extract x and y coordinates
                    x = np.tile(wave, data[0].shape[0])
                    y = np.repeat(np.arange(data[0].shape[0]), data[0].shape[1])
                    colors = data[0].flatten()
                    
                    mask = (x>linelist_SDSS[line]['lambda']/1e4 -0.001) & (x<linelist_SDSS[line]['lambda']/1e4 +0.001)
                    x = x[mask]
                    y = y[mask]
                    colors = colors[mask]
                    
                    xvelocity = wavelengthToVelocity(x,linelist_SDSS[line]['lambda']/1e4)

                    plt.subplot(12,6,cnt)
                    # Create scatter plot
                    plt.scatter(xvelocity, y, c=colors, cmap='gray', vmin=vmin, vmax=vmax, marker='s', s=20)

                    plt.xlim([-700,700])
                    # plt.xticks(np.arange(-5000,5001,2500))
                    # plt.xlim([
                    #     linelist_SDSS[line]['lambda']/1e4 -0.0008,
                    #     linelist_SDSS[line]['lambda']/1e4 +0.0008
                    #     ])
                    # plt.axvline(linelist_SDSS[line]['lambda']/1e4 ,color='r',linestyle='--')
                    plt.axvline(0 ,color='r',linestyle='--')
                    
                    plt.title(f"{line}")
                    # plt.tight_layout()
                    
                    cnt+=1
        plt.tight_layout()
        plt.savefig(f"{self.objid}_2D_zoom.png")
        plt.close()
                    

        
                
    def save(self):
        ## Save the dataProduct as pkl file
        print("Final dataproduct: ", self.dataProduct)
        save_as_pickle(self.dataProduct,self.objid+"_dataProduct.pkl")
        
        
        ## Store the spectra as csv file for easy access 
        for i in self.dataProduct.keys():
            if(i in ['sp3','sp4','sp5','sp6','sp7']):
                t = Table()
                t['lambda'] = self.dataProduct[i]['lambda']
                t['flux']   = self.dataProduct[i]['flux']
                
                t['sky_model_lambda']   = self.dataProduct[i]['sky_model_lambda']
                t['sky_model']          = self.dataProduct[i]['sky_model']
                ascii.write(t,self.objid+"_"+i+"_1D.csv",format='csv',overwrite=True)
                
                
        ## Make a report 
        cmd = "pdfjoin  --rotateoversize false %s_slitPosition.png %s-1.png %s-2.png %s-3.png %s-4.png %s-5.png %s_2D_zoom.png -o %s_report.pdf"%(self.objid,self.objid,self.objid,self.objid,self.objid,self.objid,self.objid,self.objid)
        os.system(cmd)    
              
        ## Delete existing files if any from the folder  
        cmd = "rm -r %s/*"%(self.objid)
        os.system(cmd)
        
        ## Make a folder to store the data
        cmd = "mkdir %s"%(self.objid)
        os.system(cmd)
        
        ## Move all the files to the folder
        filenames = glob.glob(self.objid+"*")
        for src in filenames:
            try:
                dst = self.objid + '/' + src
                shutil.copyfile(src, dst)
            except Exception as e:
                print(e)
                
        ## Delete existing files if any from the folder  
        cmd = "rm %s*"%(self.objid)
        os.system(cmd)
        
                
    def clean(self):
        cmd = "rm %s*"%(self.objid)        
        os.system(cmd)
        