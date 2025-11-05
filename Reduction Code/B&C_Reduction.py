"""
Title: B&C_Reduction.py
Author: Quin Aicken Davies
Date: 04/11/2025

Description: Perform reduction of B&C data using ASTR211 reduction pipeline edited to work in this file.
"""
#%%
import os # A module for communicating with the operating system e.g. commands and files
import glob # A module for searching data files
from os import killpg

import matplotlib.pyplot as plt # A plotting library
import numpy as np # Numerical Python, great for vectorised equations
import scipy # Scientific Python, great for algorithms and optimisation
import pandas as pd # DataFrames for organising data/tables
from astropy.io import fits # Astronomy Python for opening ".fits" files
from photutils.background import Background2D, MedianBackground # Fitting background surfaces to astronomical images
from copy import deepcopy # A module for copying objects
import warnings # To ignore our problems

warnings.filterwarnings('ignore', category=RuntimeWarning) #Ignores some warnings

#%matplotlib widget

#%%
# Basic 1
folder = 'Basic_Reduction_1/'  # Filenames are descriptive, exact files, minimum exposures times
#folder = '/Volumes/ASTR211/2025_Lab_Files/Photometry/Reduction_Lab/Basic_Reduction_1/'  # Filenames are descriptive, exact files, minimum exposures times
# Basic 2 - more file curation required and a variety of exposure times for flats and darks
# folder = '/Volumes/ASTR211/2025_Lab_Files/Photometry/Reduction_Lab/Basic_Reduction_2/' # Running numbers for filenames, more variety in exposure times for flats and darks

files = glob.glob(folder + '*.fit*')

darks = []
flats = []
science = []

for file in files:
    hdul = fits.open(file)
    hdr = hdul[0].header

    if "IMAGETYP" in hdr:
        if "dark" in hdr['IMAGETYP'].lower():
            darks.append(file)
        elif "flat" in hdr['IMAGETYP'].lower():
            flats.append(file)
        elif "light" in hdr['IMAGETYP'].lower():
            science.append(file)
    else:
        raise ValueError("IMAGETYP not found in header")

    print(file, hdr['IMAGETYP'], hdr['EXPTIME'])
    
#%%
listfiles_darks4science = []
listfiles_darks4flats = []
listfiles_flats = []
listfiles_science = []

darks4science_data = []
darks4flats_data = []
flats_data = []
science_data = []

for file in darks:
    hdul = fits.open(file)
    hdr = hdul[0].header
    print(file)
    if hdr['EXPTIME'] == 60:
        listfiles_darks4science.append(file)
        darks4science_data.append(hdul[0].data.copy())
    elif hdr['EXPTIME'] == 15:
        listfiles_darks4flats.append(file)
        darks4flats_data.append(hdul[0].data.copy())
    hdul.close()

for file in flats:
    hdul = fits.open(file)
    hdr = hdul[0].header
    if hdr['FILTER'] == 'r':
        listfiles_flats.append(file)
        flats_data.append(hdul[0].data.copy())
    hdul.close()

for file in science:
    hdul = fits.open(file)
    hdr = hdul[0].header
    if hdr['FILTER'] == 'r':
        if hdr['EXPTIME'] == 60:
            listfiles_science.append(file)
            science_data.append(hdul[0].data.copy())
    hdul.close()

plt.figure()
plt.imshow(darks4flats_data[0], cmap = 'gray', vmin = np.nanpercentile(darks4flats_data[0], 2), vmax = np.nanpercentile(darks4flats_data[0], 98)) # 2d plotting
#plt.axis('off') # turning off axis labels
plt.title('Example Dark for Flats')
plt.colorbar()
plt.savefig('Saved_Figures/Example_Dark4Flats.png')
plt.show()

plt.figure()
plt.imshow(darks4science_data[0], cmap = 'gray', vmin = np.nanpercentile(darks4science_data[0], 2), vmax = np.nanpercentile(darks4science_data[0], 98)) # 2d plotting
#plt.axis('off') # turning off axis labels
plt.title('Example Dark for Science')
plt.colorbar()
plt.savefig('Saved_Figures/Example_Dark4Science.png')
plt.show()

plt.figure()
plt.imshow(flats_data[0], cmap = 'gray', vmin = np.nanpercentile(flats_data[0], 2), vmax = np.nanpercentile(flats_data[0], 98)) # 2d plotting
#plt.axis('off') # turning off axis labels
plt.title('Example Flat')
plt.colorbar()
plt.savefig('Saved_Figures/Example_Flat.png')
plt.show()

plt.figure()
plt.imshow(science_data[0], cmap = 'gray', vmin = np.nanpercentile(science_data[0], 2), vmax = np.nanpercentile(science_data[0], 98)) # 2d plotting
#plt.axis('off') # turning off axis labels
plt.title('Example Raw Science')
plt.colorbar()
plt.savefig('Saved_Figures/Example_RawScience.png')
plt.show()