"""
Title: B&C_Reduction.py
Author: Quin Aicken Davies
Date: 04/11/2025

Description: Perform reduction of B&C data using parts of theASTR211 reduction pipeline 
edited to work in this file. Major adjustments include handling different exposure times for darks,
bias correction for flats, saving reduced science images to a specified directory, and more 
efficient file opening.
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

#%% Census Code
#Name of target
star = 'ZZ_Psc'
#Define Folders
main_folder = '/home/users/qai11/data/Photometery_RAW/20251030/' # Filenames are descriptive, exact files, minimum exposures times
flats_folder = '/home/users/qai11/data/Photometery_RAW/20250913/'
output_folder = '/home/users/qai11/Documents/Reduced_files/'
#Read in the flats, darks, bias and science files
dark_files = glob.glob(main_folder + '*dark*.fit*')
flat_files = glob.glob(flats_folder + '*flat**g*.fit*')
darks_for_flats_5 = glob.glob(flats_folder + '*dark*d5*.fit*')
darks_for_flats_1 = glob.glob(flats_folder + '*dark*_1.*fit*')
bias_files = glob.glob(main_folder + '*bias*.fit*')
science_files = glob.glob(main_folder + f'*{star}**g*.fit*')

#Check how many files of each type we have
print(f'Number of dark files: {len(dark_files)}')
print(f'Number of flat files: {len(flat_files)}')
print(f'Number of bias files: {len(bias_files)}')
print(f'Number of science files: {len(science_files)}')
print(f'Number of darks for flats files: {len(darks_for_flats_5)}')

for file in flat_files:
    hdul = fits.open(file)
    hdr = hdul[0].header
    print(f'EXPTIME: {hdr["EXPTIME"]}, FILTER: {hdr["FILTER"]}, Readout: {hdr["READOUTM"]}, CCDTEMP: {hdr["CCD-TEMP"]}')
    hdul.close()
print("flat Darks")
for file in darks_for_flats_5:
    hdul = fits.open(file)
    hdr = hdul[0].header
    print(f'EXPTIME: {hdr["EXPTIME"]}, Readout: {hdr["READOUTM"]}, CCDTEMP: {hdr["CCD-TEMP"]}')
    hdul.close()
for file in darks_for_flats_1:
    hdul = fits.open(file)
    hdr = hdul[0].header
    print(f'EXPTIME: {hdr["EXPTIME"]}, Readout: {hdr["READOUTM"]}, CCDTEMP: {hdr["CCD-TEMP"]}')
    hdul.close()
print("Science Darks")
science_darks = []
for file in dark_files:
    hdul = fits.open(file)
    hdr = hdul[0].header
    print(f'EXPTIME: {hdr["EXPTIME"]}, Readout: {hdr["READOUTM"]}, CCDTEMP: {hdr["CCD-TEMP"]}')
    if hdr["CCD-TEMP"] == -40:
        science_darks.append(file)
    hdul.close()

#%% Separate the files into relevant categories by opening them and
# sorting the infromation based on header information
listfiles_darks4science = []
listfiles_darks4flats_5 = []
listfiles_flats = []
listfiles_science = []
listfiles_science_bias = []
listfiles_darks4flats_1 = []

darks4science_data = []
darks4flats_data_5 = []
flats_data = []
science_data = []
science_hdr = []
bias_data = []
darks4flats_data_1 = []
flats_headers = []

# Combine file processing into a single loop for each category
for file in dark_files + darks_for_flats_5 + darks_for_flats_1 + flat_files + science_files+ bias_files:
    hdul = fits.open(file)
    hdr = hdul[0].header
    data = hdul[0].data.copy()
    
    if file in dark_files and hdr['EXPTIME'] == 40:
        listfiles_darks4science.append(file)
        darks4science_data.append(data)
    elif file in darks_for_flats_5 and hdr['EXPTIME'] == 0.5:
        listfiles_darks4flats_5.append(file)
        darks4flats_data_5.append(data)
    elif file in darks_for_flats_1 and hdr['EXPTIME'] == 1:
        listfiles_darks4flats_1.append(file)
        darks4flats_data_1.append(data)
    elif file in flat_files and hdr['FILTER'] == 'g':
        if np.nanmedian(data) > 20000:
            listfiles_flats.append(file)
            flats_data.append(data)
            flats_headers.append(hdr)
    elif file in science_files and hdr['FILTER'] == 'g' and hdr['EXPTIME'] == 40:
        listfiles_science.append(file)
        science_data.append(data)
        science_hdr.append(hdr)  # Store the header of the last science file for later use
    elif file in bias_files:
        listfiles_science_bias.append(file)
        bias_data.append(data)
    hdul.close()

plt.figure()
plt.imshow(darks4flats_data_5[0], cmap = 'gray', vmin = np.nanpercentile(darks4flats_data_5[0], 2), vmax = np.nanpercentile(darks4flats_data_5[0], 98)) # 2d plotting
plt.title('Example Dark for Flats 0.5 Seconds')
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(darks4flats_data_1[0], cmap = 'gray', vmin = np.nanpercentile(darks4flats_data_1[0], 2), vmax = np.nanpercentile(darks4flats_data_1[0], 98)) # 2d plotting
plt.title('Example Dark for Flats 1 Seconds')
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(darks4science_data[0], cmap = 'gray', vmin = np.nanpercentile(darks4science_data[0], 2), vmax = np.nanpercentile(darks4science_data[0], 98)) # 2d plotting
plt.title('Example Dark for Science')
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(flats_data[0], cmap = 'gray', vmin = np.nanpercentile(flats_data[0], 2), vmax = np.nanpercentile(flats_data[0], 98)) # 2d plotting
plt.title('Example Flat')
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(science_data[0], cmap = 'gray', vmin = np.nanpercentile(science_data[0], 2), vmax = np.nanpercentile(science_data[0], 98)) # 2d plotting
plt.title('Example Raw Science')
plt.colorbar()
plt.show()

# %% Create Master Darks
master_dark_science = np.median(np.array(darks4science_data), axis=0)
master_dark_flats_5 = np.median(np.array(darks4flats_data_5), axis=0)
master_dark_flats_1 = np.median(np.array(darks4flats_data_1), axis=0)
#Save master dark science
hdu = fits.PrimaryHDU(master_dark_science)
hdu.writeto(output_folder + 'master_dark_science.fits', overwrite=True)

plt.figure()
plt.imshow(master_dark_science, cmap = 'gray', vmin = np.nanpercentile(master_dark_science, 2), vmax = np.nanpercentile(master_dark_science, 98)) # 2d plotting
plt.title('Master Dark for Science')
plt.colorbar()
plt.show()
plt.figure()
plt.imshow(master_dark_flats_5, cmap = 'gray', vmin = np.nanpercentile(master_dark_flats_5, 2), vmax = np.nanpercentile(master_dark_flats_5, 98)) # 2d plotting
plt.title('Master Dark for Flats 0.5 Seconds')
plt.colorbar()
plt.show()
plt.figure()
plt.imshow(master_dark_flats_1, cmap = 'gray', vmin = np.nanpercentile(master_dark_flats_1, 2), vmax = np.nanpercentile(master_dark_flats_1, 98)) # 2d plotting
plt.title('Master Dark for Flats 1 Second')
plt.colorbar()
plt.show()

# %% Create Master Flat
corrected_flats_05 = []
corrected_flats_1 = []
average_bias = np.nanmedian(bias_data)  # Calculate average bias

for flat, hdr in zip(flats_data, flats_headers):
    try:
        if hdr['EXPTIME'] == 0.5:
            corrected = flat - master_dark_flats_5 - average_bias
            corrected_flats_05.append(corrected / np.nanmedian(corrected))
        elif hdr['EXPTIME'] == 1.0:
            corrected = flat - master_dark_flats_1 - average_bias
            corrected_flats_1.append(corrected / np.nanmedian(corrected))
    except Exception as e:
        print(f"Skipping flat due to error: {e}")
        continue

master_flat_05 = np.median(np.array(corrected_flats_05), axis=0)
master_flat_1 = np.median(np.array(corrected_flats_1), axis=0)
master_flat_combined = np.median(np.array([master_flat_05, master_flat_1]), axis=0)
#Save the master flat
hdu = fits.PrimaryHDU(master_flat_combined)
hdu.writeto(output_folder + 'master_flat_combined.fits', overwrite=True)


#plot the master flats
plt.figure()
plt.imshow(master_flat_05, cmap = 'gray', vmin = np.nanpercentile(master_flat_05, 2), vmax = np.nanpercentile(master_flat_05, 98)) # 2d plotting
plt.title('Master Flat 0.5 Seconds')
plt.colorbar()
plt.show()
plt.figure()
plt.imshow(master_flat_1, cmap = 'gray', vmin = np.nanpercentile(master_flat_1, 2), vmax = np.nanpercentile(master_flat_1, 98)) # 2d plotting
plt.title('Master Flat 1 Seconds')
plt.colorbar()
plt.show()
plt.figure()
plt.imshow(master_flat_combined, cmap = 'gray', vmin = np.nanpercentile(master_flat_combined, 2), vmax = np.nanpercentile(master_flat_combined, 98)) # 2d plotting
plt.title('Combined Master Flat')
plt.colorbar() 
plt.show()

# %% Reduce Science Images
reduced_science_images = []
for science in science_data:
    corrected_science = (science - master_dark_science) / master_flat_combined
    reduced_science_images.append(corrected_science)
    
plt.figure()
plt.imshow(science_data[0], cmap = 'gray', vmin = np.nanpercentile(science_data[0], 2), vmax = np.nanpercentile(science_data[0], 98)) # 2d plotting
plt.title('Example Raw Science Image')
plt.colorbar()
plt.show()
plt.figure()
plt.imshow(reduced_science_images[0], cmap = 'gray', vmin = np.nanpercentile(reduced_science_images[0], 2), vmax = np.nanpercentile(reduced_science_images[0], 98)) # 2d plotting
plt.title('Example Reduced Science Image')
plt.colorbar()
plt.show()
# %% Save Reduced Images
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
for i, (reduced_image, hdr) in enumerate(zip(reduced_science_images, science_hdr)):
    hdu = fits.PrimaryHDU(reduced_image, header=fits.Header(hdr))
    # hdu = fits.HDUList([hdu])
    output_filename = os.path.join(output_folder, f'{star}_reduced_{i+1}.fits')
    hdu.writeto(output_filename, overwrite=True)
    print(f'Saved: {output_filename}')
    
    

# %%
