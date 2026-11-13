import glob
from scipy.interpolate import UnivariateSpline 
from scipy.signal import medfilt
from astropy.io import ascii, fits
import matplotlib.pyplot as plt 
import numpy as np
from lmfit import Model
from ISMgas.visualization.fits import ScaleImage
from ISMgas.SupportingFunctions import load_pickle, interpolateData, save_as_pickle
from numpy import exp, loadtxt, pi, sqrt
def gaussian(x, amp, cen, wid):
    """1-d gaussian: gaussian(x, amp, cen, wid)"""
    return (amp / (sqrt(2*pi) * wid)) * exp(-(x-cen)**2 / (2*wid**2))



class mageRedux:
    def __init__(self, objid, folder, extractWindow, stdstar):
        
        #Init  standard star #
        self.stdstarDict= {
            'ltt3864' : {'specFile': "standards/fltt3864.dat"} ## Insert others
        }   
        self.stdstar = stdstar
        self.stdstarSpec = ascii.read(self.stdstarDict[self.stdstar]['specFile'], names=["wave", "FLAM16", "flux", "bins"])
        #Smooth
        self.stdstarSpec['FLAM16'] = medfilt(self.stdstarSpec['FLAM16'], kernel_size=11)
        

        plt.plot(
            self.stdstarSpec['wave'],
            self.stdstarSpec['FLAM16'],
            color='black',
            drawstyle='steps-mid'
        )

    
        self.objid = objid
        self.folder = folder

        self.files = glob.glob(self.folder + "*-0*sum.fits")
        self.extractWindowMin = extractWindow['min']
        self.extractWindowMax = extractWindow['max']



    
    def extractAndPlot(self,  ylim=[0,1200], plotting=True, standardstar=False, clip_threshold=0.3, scaleError = 1):
    
        extracted_spectra = {}
        objid = self.objid

        
        for order,file in enumerate(sorted(self.files)):
            hdr0 = fits.getheader(file)
            data = fits.getdata(file)
            error = fits.getdata(file.replace("sum.fits", "sumsig.fits"))
            
            wave0 = (np.arange(hdr0['NAXIS1']) - hdr0['CRPIX1'] + 1) * hdr0['CD1_1'] + hdr0['CRVAL1']
    
            if(plotting):
                plt.figure(figsize=(12,12))
                ScaleImage(data).plot()
                plt.gca().set_aspect(aspect=12)
                plt.show()
            
    
            data_mask = (data==0) ## This traces how the slit changes along the dispersion direction
            data_mask.astype(int)
            trace2D = data*0
            
            ## First pass at getting the offsets
            offsets = []
            for idx in range(data.shape[1]):
                offset = 0 
            
                dataTop = data_mask[:round(hdr0['OBJPOS']), idx] 
                dataBottom = data_mask[round(hdr0['OBJPOS']):, idx]
        
                offsetTop = np.nansum(dataTop,axis=0) ## in pixels
                offsetBottom = np.nansum(dataBottom,axis=0) ## in pixels
    
                offsets.append(round(hdr0['OBJPOS']) + offsetTop- offsetBottom)
                trace2D[
                    round(hdr0['OBJPOS'])- self.extractWindowMin + (offsetTop- offsetBottom): round(hdr0['OBJPOS'])+ self.extractWindowMax+1 + (offsetTop- offsetBottom),
                    idx
                    ] =1 
    
            idxs = list(range(data.shape[1]))
    
            # Use a cubic fit to get a functional form for the offsets
            deg = 3  # cubic fit
            
            # polynomial fit
            coeffs = np.polyfit(idxs, offsets, deg)
            poly = np.poly1d(coeffs)
            
            # evaluate on dense grid
            offsets_fit = poly(idxs)
            
            ## Final form of the extracted trace
            trace2D = data*0
            for idx in range(data.shape[1]):
                data_extract = data[:,idx].copy()
                idx_extract = np.asarray(list(range(len(data_extract))))

                ## Clean 
                mask = ~np.isnan(data_extract)
                data_extract = data_extract[mask]
                idx_extract  = idx_extract[mask]

                if(standardstar):
                    ## Fit gaussian to find the exact center 
                    model = Model(gaussian)

                    params = model.make_params() 

                    params['cen'].set(value = offsets_fit[idx])
                    params['wid'].set(value = 5,min=1,max=100)
                    params['amp'].set(value = 5,min=0.1,max=100)

                    result = model.fit(data_extract, params, x=idx_extract)

                    # result.plot()
                    # plt.close()
                    # plt.figure()
                    # plt.plot(idx_extract, data_extract)
                    # plt.show()
                    # print(result.params['cen'].value,offsets_fit[idx] )

                    trace2D[
                        round(result.params['cen'].value) - self.extractWindowMin: 
                        round(result.params['cen'].value) + self.extractWindowMax+1,
                        idx] =1 

                else:
    
                    trace2D[
                        round(offsets_fit[idx]) - self.extractWindowMin: 
                        round(offsets_fit[idx]) + self.extractWindowMax+1,
                        idx] =1 
    
            if(plotting):
                plt.figure(figsize=(12,12))
                ScaleImage(data).plot() 
                plt.imshow(trace2D, origin='lower', cmap = 'rainbow', alpha = 0.3)       
                plt.gca().set_aspect(aspect=12)
                plt.scatter(idxs, offsets, marker='^')
                plt.plot(idxs, offsets_fit, color='red', lw=3)
                plt.show()
    
                plt.figure(figsize=(12,12))
                ScaleImage(data).plot()
                plt.imshow(trace2D, origin='lower', cmap = 'rainbow', alpha = 0.3)       
                plt.gca().set_aspect(aspect=12)
                plt.show()
    

            ## Extract spectra
            extractWave = wave0
            extractSpec = np.nansum(data*trace2D, axis=0) 
            extractVariation = np.nanstd(data*trace2D, axis=0)
            extractSNR = extractSpec/extractVariation ## SNR
  
            extractErr  = scaleError* np.sqrt(np.nansum((error**2)*trace2D, axis=0)) # in quadrature



            if(plotting):
                plt.figure(figsize=(12,7))
                plt.plot(
                    extractWave,
                    extractSpec,
                    color='black',
                    drawstyle='steps-mid'
                )
                plt.fill_between(
                    extractWave,
                    extractSpec - extractErr * 1,
                    extractSpec + extractErr * 1,
                    alpha       = 0.25,
                    color='gray'
                )
                plt.ylim([0,None])
                plt.xlim([wave0[0], wave0[-1]])
              
    
            if(standardstar):
    
                fooWave = extractWave
                fooFlux  = extractSpec
                fooError = extractErr
    
                from scipy.interpolate import interp1d
                ## Flux calibrate
                f = interp1d(self.stdstarSpec['wave'], self.stdstarSpec['FLAM16'], fill_value='extrapolate')
                interpolate_flux = f(fooWave)


                from scipy.ndimage import gaussian_filter
                extractSpec = gaussian_filter(fooFlux, sigma=100)

                ## Clip the sensitivity function at the edges 
                threshold = clip_threshold # 20% of peak flux

                mask_thresh = extractSpec < threshold * np.nanmax(extractSpec)

                extractWave[mask_thresh] = np.nan
                extractSpec[mask_thresh] = np.nan
                extractErr[mask_thresh] = np.nan


                plt.plot(
                        extractWave,
                        extractSpec,
                        color='red',
                        drawstyle='steps-mid'
                    )

                extracted_spectra[order]={
                    'wave': extractWave,
                    'flux': extractSpec, 
                    'error':  extractErr,
                    'sensitivity': interpolate_flux/extractSpec
                }
            
            else:
                extracted_spectra[order]= {
                    'wave': extractWave,
                    'flux': extractSpec,
                    'error' : extractErr
                }

            plt.show()
        # Save the extracted spectra 
        save_as_pickle(extracted_spectra, objid + "_extracted.pkl")
    
    def correctSensitivity(self, kernel_size_ang=50, plotting=True,zs=0, ylim=[0,None], sigma_threshold=1):
        """
        Given a object extracted and a standard star extracted pickle files, 
        flux calibrate the data.
        """
        objid = self.objid 
        object_spec = load_pickle(self.objid + "_extracted.pkl")
        standard_spec = load_pickle(self.stdstar + "_extracted.pkl")
    
        corrected_spec = {}
    
        common_orders = set(object_spec.keys()) & set(standard_spec.keys())
        
        if(plotting):
            plt.figure(figsize=(12,5))
            for order in sorted(common_orders):
                std_wave = standard_spec[order]["wave"]
                std_flux = standard_spec[order]["flux"]
                std_inv_sensitivity = standard_spec[order]["sensitivity"]
                plt.plot(std_wave, 1/std_inv_sensitivity)

            plt.ylim([0,1000])
    
        for order in sorted(common_orders):
    
            # Object
            obj_wave = object_spec[order]["wave"]
            obj_flux = object_spec[order]["flux"]
            obj_error = object_spec[order]["error"]
    
            # Standard
            std_wave = standard_spec[order]["wave"]
            std_flux = standard_spec[order]["flux"]
            std_inv_sensitivity = standard_spec[order]["sensitivity"]
                    
            # Determine overlapping wavelength range
            mask = (
                (obj_wave >= np.nanmin(std_wave)) &
                (obj_wave <= np.nanmax(std_wave))
            )
            # mask = ~np.isnan(std_wave)
            obj_wave = obj_wave[mask]
            obj_flux = obj_flux[mask]
            obj_error = obj_error[mask]


            std_invsens_interp = interpolateData(x=std_wave,y=std_inv_sensitivity,xnew=obj_wave, fill_value='extrapolate')
            # std_invsens_interp = medfilt(std_invsens_interp, kernel_size=101) ## Smooth out any artifacts
                        
            corrected_flux = obj_flux * std_invsens_interp
            corrected_error = obj_error * std_invsens_interp

            ## plot inv_sens curve
            plt.figure(figsize=(12,5))
            plt.plot(obj_wave,corrected_flux,color='black',drawstyle='steps-mid' )


            # plt.plot(obj_wave,corrected_flux,color='red',drawstyle='steps-mid', label='clipped' )
            plt.ylim([0,np.nanmedian(corrected_flux) + np.nanstd(corrected_flux)])
            # plt.xlim([obj_wave[0], obj_wave[-1]])
            plt.show()

            corrected_spec[order] = {
                "wave": obj_wave,
                "flux": corrected_flux,
                "error":corrected_error,
                "inv_sensitivity": std_invsens_interp,
            }
    
            if(plotting):
  
            
                plt.figure(figsize=(12,7))
                plt.plot(
                    obj_wave,
                    corrected_flux,
                    color='black',
                    drawstyle='steps-mid'
                )
    
                plt.fill_between(
                    obj_wave,
                    corrected_flux - corrected_error * 1,
                    corrected_flux + corrected_error * 1,
                    alpha       = 0.25,
                    color='gray'
                )
                plt.ylim([0,np.nanmedian(corrected_flux) + np.nanstd(corrected_flux)])
                plt.xlim([obj_wave[0], obj_wave[-1]])
                plt.show() 

        # Save 
        save_as_pickle(corrected_spec, self.objid + "_extracted_sc.pkl")
            
        
        
    def combine_echelle_orders(self,
                            wave_grid=None,
                            use_sensitivity=True,
                            min_weight=0,
                            combine_method='weighted'):
        """
        This code block is inspired from IFAF's scombine
        """
        orders = load_pickle(self.objid + "_extracted_sc.pkl")

        if wave_grid is None:
            dw = np.min([
                np.nanmedian(np.diff(order['wave']))
                for order in orders.values()
            ])

            wmin = min(np.nanmin(order['wave']) for order in orders.values())
            wmax = max(np.nanmax(order['wave']) for order in orders.values())

            wave_grid = np.arange(wmin, wmax + dw, dw)

        # ----------------------------------------------------------
        # Arrays used for combination
        # ----------------------------------------------------------
        numerator = np.zeros_like(wave_grid, dtype=float)
        denominator = np.zeros_like(wave_grid, dtype=float)

        # For simple average
        flux_sum = np.zeros_like(wave_grid, dtype=float)
        flux_count = np.zeros_like(wave_grid, dtype=int)

        # ----------------------------------------------------------
        # Combine orders
        # ----------------------------------------------------------
        for order in orders.values():

            wave = order['wave']
            flux = order['flux']
            err = order['error']

            mask = (
                np.isfinite(wave) &
                np.isfinite(flux) &
                np.isfinite(err) &
                (err > 0)
            )

            if np.sum(mask) < 2:
                continue

            wave = wave[mask]
            flux = flux[mask]
            err = err[mask]

            # ------------------------------------------------------
            # Interpolate onto common wavelength grid
            # ------------------------------------------------------
            flux_i = np.interp(
                wave_grid, wave, flux,
                left=np.nan, right=np.nan
            )

            err_i = np.interp(
                wave_grid, wave, err,
                left=np.nan, right=np.nan
            )

            # ------------------------------------------------------
            # Simple average
            # ------------------------------------------------------
            if combine_method.lower() in ['average', 'mean', 'simple']:

                good = (
                    np.isfinite(flux_i) &
                    np.isfinite(err_i) &
                    (err_i > 0)
                )

                flux_sum[good] += flux_i[good]
                flux_count[good] += 1

            # ------------------------------------------------------
            # Weighted average -- IRAF/scombine-like
            # ------------------------------------------------------
            elif combine_method.lower() in ['weighted', 'ivar']:

                weight = 1.0 / err_i**2

                weight[~np.isfinite(weight)] = 0
                weight[weight < min_weight] = 0

                good = np.isfinite(flux_i) & (weight > 0)

                numerator[good] += flux_i[good] * weight[good]
                denominator[good] += weight[good]

            else:
                raise ValueError(
                    "combine_method must be 'average' or 'weighted'"
                )

        # ----------------------------------------------------------
        # Construct combined spectrum
        # ----------------------------------------------------------
        combined_flux = np.full_like(wave_grid, np.nan)
        combined_err = np.full_like(wave_grid, np.nan)

        if combine_method.lower() in ['average', 'mean', 'simple']:

            good = flux_count > 0

            combined_flux[good] = (
                flux_sum[good] / flux_count[good]
            )

            # Error on the mean.
            #
            # Since each order has its own error, propagate the
            # individual errors rather than assuming identical errors.
            #
            # This requires another pass through the orders.
            error_sum = np.zeros_like(wave_grid, dtype=float)

            for order in orders.values():

                wave = order['wave']
                flux = order['flux']
                err = order['error']

                mask = (
                    np.isfinite(wave) &
                    np.isfinite(flux) &
                    np.isfinite(err) &
                    (err > 0)
                )

                if np.sum(mask) < 2:
                    continue

                wave = wave[mask]
                flux = flux[mask]
                err = err[mask]

                flux_i = np.interp(
                    wave_grid, wave, flux,
                    left=np.nan, right=np.nan
                )

                err_i = np.interp(
                    wave_grid, wave, err,
                    left=np.nan, right=np.nan
                )

                good = (
                    np.isfinite(flux_i) &
                    np.isfinite(err_i) &
                    (err_i > 0)
                )

                error_sum[good] += err_i[good]**2

                [good] = (
                np.sqrt(error_sum[good]) / flux_count[good]
            )

        elif combine_method.lower() in ['weighted', 'ivar']:

            good = denominator > 0

            combined_flux[good] = (
                numerator[good] / denominator[good]
            )

            combined_err[good] = np.sqrt(
                1.0 / denominator[good]
            )

        extracted_spec = {
            'wave': wave_grid,
            'flux': combined_flux,
            'error': combined_err
        }

        save_as_pickle(
            extracted_spec,
            self.objid + "_extracted_sc_ec.pkl"
        )

        return wave_grid, combined_flux, combined_err, denominator
