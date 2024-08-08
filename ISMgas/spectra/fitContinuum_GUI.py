### The original source code was written by course written by Tom Aldcroft, Tom Robitaille, 
### Brian Refsdal, Gus Muench (Copyright 2011, Smithsonian Astrophysical Observatory) and 
### released under a Creative Commons Attribution 3.0 License.
### Source : https://python4esac.github.io/index.html 
### I (KVGC) have since then modified it to work with z~2 spectra.

import pylab as plt
import numpy as np
from scipy.interpolate import splrep,splev
from astropy.io import ascii
import sys
import os

def onclick(event):
    # when none of the toolbar buttons is activated and the user clicks in the
    # plot somewhere, compute the median value of the spectrum in a 10angstrom
    # window around the x-coordinate of the clicked point. The y coordinate
    # of the clicked point is not important. Make sure the continuum points
    # `feel` it when it gets clicked, set the `feel-radius` (picker) to 5 points
    toolbar = plt.get_current_fig_manager().toolbar
    if event.button==1 and toolbar.mode=='':
        window = ((event.xdata-1)<=wave) & (wave<=(event.xdata))
        y = np.median(flux[window])
        plt.plot(event.xdata,y,'rs',ms=10,picker=5,label='cont_pnt')
    plt.draw()

def onpick(event):
    # when the user clicks right on a continuum point, remove it
    if event.mouseevent.button==3:
        if hasattr(event.artist,'get_label') and event.artist.get_label()=='cont_pnt':
            event.artist.remove()

def ontype(event):
    # when the user hits enter:
    # 1. Cycle through the artists in the current axes. If it is a continuum
    #    point, remember its coordinates. If it is the fitted continuum from the
    #    previous step, remove it
    # 2. sort the continuum-point-array according to the x-values
    # 3. fit a spline and evaluate it in the wavelength points
    # 4. plot the continuum
    if event.key=='enter':
        cont_pnt_coord = []
        for artist in plt.gca().get_children():
            if hasattr(artist,'get_label') and artist.get_label()=='cont_pnt':
                cont_pnt_coord.append(artist.get_data())
            elif hasattr(artist,'get_label') and artist.get_label()=='continuum':
                artist.remove()
        cont_pnt_coord = np.array(cont_pnt_coord)[...,0]
        sort_array = np.argsort(cont_pnt_coord[:,0])
        x,y = cont_pnt_coord[sort_array].T
        np.savetxt("chosenPoints.nspec",  cont_pnt_coord[sort_array])
        spline = splrep(x,y,k=3)
        continuum = splev(wave,spline)
        plt.plot(wave,continuum,'r-',lw=2,label='continuum', drawstyle='steps-mid')

    # when the user hits 'n' and a spline-continuum is fitted, normalise the
    # spectrum
    elif event.key=='n':
        continuum = None
        for artist in plt.gca().get_children():
            if hasattr(artist,'get_label') and artist.get_label()=='continuum':
                continuum = artist.get_data()[1]
                break
        np.savetxt('continuum'+'.nspec',np.array([wave,continuum]).T)
        if continuum is not None:
            plt.cla()
            plt.plot(wave,flux/continuum,'k-',label='normalised',drawstyle='steps-mid')

    # when the user hits 'r': clear the axes and plot the original spectrum
    elif event.key=='r':
        plt.cla()
        plt.plot(wave,flux,'k-')

    # when the user hits 'w': if the normalised spectrum exists, write it to a
    # file.
    elif event.key=='w':
        for artist in plt.gca().get_children():
            if hasattr(artist,'get_label') and artist.get_label()=='normalised':
                data = np.array(artist.get_data())
                np.savetxt('spec'+'.nspec',data.T)
                print('Saved to file')
                break
    plt.draw()


if __name__ == "__main__":
    # Get the filename of the spectrum from the command line, and plot it
    #filename = sys.argv[1]
    #wave,flux = np.loadtxt(filename).T
    zs =1.86523

    t_bright = ascii.read("/home/keerthi/Dropbox/kvgc/ThisCodeWorks-main_OLD/KCWI/CSWA13/masks/mainarc-brighregionsonly")
    fitting_profile     = t_bright['col2']
    fitting_wav      = np.arange(3229,3229+len(fitting_profile),1)  
    fitting_wav      = fitting_wav/(1+zs)
    wave,flux = [fitting_wav, fitting_profile]
    spectrum, = plt.plot(wave,flux,'k-',label='spectrum',drawstyle='steps-mid')
#    plt.title(filename)

    # Connect the different functions to the different events
    plt.gcf().canvas.mpl_connect('key_press_event',ontype)
    plt.gcf().canvas.mpl_connect('button_press_event',onclick)
    plt.gcf().canvas.mpl_connect('pick_event',onpick)
    plt.show() # show the window
