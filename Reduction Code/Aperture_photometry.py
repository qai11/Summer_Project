"""
Title: Aperture_photometry.py
Author: Quin Aicken Davies
Date: 05/11/2025

Description: Perform aperture photometry on astronomical images using photutils. This script
reads in the fits files from B&C_Reduction.py, performs aperture photometry on specified stars,
and outputs the results to a CSV file.
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
import glob
from copy import deepcopy

from scipy.optimize import curve_fit

from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.time import Time

from photutils.aperture import CircularAperture, aperture_photometry, CircularAnnulus
from photutils.aperture import ApertureStats
from photutils.detection import DAOStarFinder
from photutils.background import Background2D, MedianBackground
from astropy.utils.exceptions import AstropyWarning
#PSF modules
from photutils.psf import extract_stars, EPSFStars, EPSFBuilder, EPSFModel
from photutils.psf import PSFPhotometry, IterativePSFPhotometry

import warnings
from astropy.coordinates import SkyCoord
warnings.filterwarnings('ignore', category=AstropyWarning, message=".*'datfix' made the change.*")


# %matplotlib inline
#%%
#Locate the target in the image
star = 'ZZ_Psc'
# #Define Folders
main_folder = '/home/users/qai11/Documents/Reduced_files/'
output_folder = '/home/users/qai11/Documents/Reduced_files/'
#Read in the reduced science files
science_files = glob.glob(os.path.join(main_folder, f'{star}_reduced_*.fits'))
print(f'Number of reduced science files: {len(science_files)}')

#%% Find Sources
def find_sources(data, x, y):
    """
    FROM ASTR211 LAB 2
    
    Find sources in the image data using DAOStarFinder.

    Parameters:
    data (2D array): The image data.
    x (float): Approximate x-coordinate of the target star.
    y (float): Approximate y-coordinate of the target star.

    Returns:
    sources (Table): Table of detected sources.
    """
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)  # Adjust sigma as needed.
    print(mean,median,std)

    # Initialize DAOStarFinder with the background statistics
    daofind = DAOStarFinder(fwhm=3.0, threshold=10.0*std)  # Adjust fwhm and threshold as needed
    sources = daofind(data) # Find sources in the image

    # Filter sources to find the target
    search_radius = 3
    filtered_sources = sources[(np.abs(sources['xcentroid'] - x) <= search_radius) & (np.abs(sources['ycentroid'] - y) <= search_radius)]

    positions = np.zeros((len(filtered_sources), 2))
    positions[:,0] = filtered_sources['xcentroid'].value
    positions[:,1] = filtered_sources['ycentroid'].value
    
    return positions


#%% Perform WCS transformation to find pixel coordinates
def world_to_pixel(ra, dec, wcs):
    """
    Convert world coordinates (RA, Dec) to pixel coordinates using WCS.

    Parameters:
    ra (float): Right Ascension in degrees.
    dec (float): Declination in degrees.
    wcs (WCS): World Coordinate System object.

    Returns:
    (x, y): Pixel coordinates.
    """
    sky_coord = SkyCoord(ra=ra, dec=dec, unit='deg')
    x, y = wcs.world_to_pixel(sky_coord)
    return x, y

#%% Function to perform aperture photometry
def example_background(data):
    bkg_estimator = Background2D(data, (50, 50), filter_size=(7, 7), bkg_estimator=MedianBackground())
    median_background = bkg_estimator.background
    return median_background

# Define a function to perform aperture photometry on a given image
def perform_aperture_photometry(image_data, positions, aperture_radius=5, annulus_radii=(7, 10)):
    """
    Perform aperture photometry on the given image data at specified positions.

    Parameters:
    image_data (2D array): The image data to perform photometry on.
    positions (list of tuples): List of (x, y) positions for the stars.
    aperture_radius (float): Radius of the circular aperture.
    annulus_radii (tuple): Inner and outer radii of the background annulus.

    Returns:
    phot_table (Table): Table containing the photometry results.
    """
    apertures = CircularAperture(positions, r=aperture_radius)
    annuli = CircularAnnulus(positions, r_in=annulus_radii[0], r_out=annulus_radii[1])
    
    # Perform aperture photometry
    process_data = image_data - example_background(image_data)
    phot_table = aperture_photometry(process_data, apertures)
    
    # Calculate background statistics
    bkg_stats = ApertureStats(process_data, annuli)
    bkg_mean = bkg_stats.mean
    
    # Subtract background from aperture sum
    phot_table['aperture_sum_bkgsub'] = phot_table['aperture_sum'] - bkg_mean * apertures.area
    
    return phot_table


#%% Process each science file
#apply WCS to get pixel coordinates for all science files
wcs_hdul = fits.open(main_folder+'wcs.fits')
wcs = WCS(wcs_hdul[0].header)
wcs_hdul.close()
# print(wcs)
results = []
for file in science_files:
    hdul = fits.open(file)
    hdr = hdul[0].header
    data = hdul[0].data.copy()
    #Find sources in the image using find sources function

    #Taget star coordinates (example coordinates, replace with actual star coordinates)
    ra = (23 * 15) + (28 / 4) + (46.95 / 240)  # Convert RA from hours, minutes, seconds to degrees
    dec = (5) + (14 / 60) + (47.32 / 3600)  # Convert Dec from degrees, arcminutes, arcseconds to degrees
    #Convert ra and dec to degrees
    print(f'Target coordinates (deg): RA={ra}, Dec={dec}')
    star_positions = [wcs.all_world2pix(ra, dec, 0)]
    
    # Perform aperture photometry
    phot_table = perform_aperture_photometry(data, star_positions)
    
    # Add time information
    time_obs = Time(hdr['DATE-OBS']).jd
    # print(time_obs)
    phot_table['time'] = time_obs
    
    results.append(phot_table)
    hdul.close()
#%%
plt.figure()
plt.imshow(data, cmap='gray', origin='lower', vmin=np.percentile(data, 5), vmax=np.percentile(data, 95))
plt.scatter(star_positions[0][0], star_positions[0][1], color='red', marker='x')
plt.show()
#%% Combine results and save to CSV
all_results = pd.concat([r.to_pandas() for r in results], ignore_index=True)
all_results = all_results[['time', 'aperture_sum_bkgsub']]  # Reorder columns
all_results = all_results[all_results['aperture_sum_bkgsub'] >= 0]  # Remove values less than zero
all_results.to_csv(output_folder + f'{star}_aperture_photometry_results.tsv', sep='\t', index=False, header=None)
print(f'Aperture photometry results saved to {star}_aperture_photometry_results.tsv')

#%% Example plot of the first star's aperture photometry results
plt.figure()
plt.scatter(all_results['time'], all_results['aperture_sum_bkgsub'])
plt.xlabel('Julian Date')
plt.ylabel('Flux (ADU)')
plt.title(f'Aperture Photometry Results for {star}')
plt.ylim(np.nanpercentile(all_results['aperture_sum_bkgsub'], 5), np.nanpercentile(all_results['aperture_sum_bkgsub'], 95)*1.1)
plt.show()

# %%
