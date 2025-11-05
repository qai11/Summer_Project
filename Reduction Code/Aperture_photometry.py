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

from astropy.utils.exceptions import AstropyWarning

import warnings
from astropy.coordinates import SkyCoord
warnings.filterwarnings('ignore', category=AstropyWarning, message=".*'datfix' made the change.*")


# %matplotlib inline
#%%
#Locate the target in the image
star = 'ZZ_Psc'
# #Define Folders
main_folder = '/home/users/qai11/Documents/Summer_Project_2025/Reduced_files/'
output_folder = '/home/users/qai11/Documents/Summer_Project_2025/Reduction Code'
#Read in the reduced science files
science_files = glob.glob(os.path.join(main_folder, f'{star}_reduced_*.fits'))
print(f'Number of reduced science files: {len(science_files)}')

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
    phot_table = aperture_photometry(image_data, apertures)
    
    # Calculate background statistics
    bkg_stats = ApertureStats(image_data, annuli)
    bkg_mean = bkg_stats.mean
    
    # Subtract background from aperture sum
    phot_table['aperture_sum_bkgsub'] = phot_table['aperture_sum'] - bkg_mean * apertures.area
    
    return phot_table
#%% Process each science file
#apply WCS to get pixel coordinates for all science files
results = []
for file in science_files:
    hdul = fits.open(file)
    hdr = hdul[0].header
    data = hdul[0].data.copy()
    
    # Get WCS information
    wcs = WCS(hdr)
    
    #Taget star coordinates (example coordinates, replace with actual star coordinates)
    ra = (23 * 15) + (28 / 4) + (46.95 / 240)  # Convert RA from hours, minutes, seconds to degrees
    dec = (5 * 15) + (14 / 4) + (47.32 / 240)  # Convert Dec from degrees, minutes, seconds to degrees
    #Convert ra and dec to degrees
    print(f'Target coordinates (deg): RA={ra}, Dec={dec}')
    star_positions = [wcs.all_world2pix(ra, dec, 0)]
    
    # Perform aperture photometry
    phot_table = perform_aperture_photometry(data, star_positions)
    
    # Add time information
    time_obs = Time(hdr['DATE-OBS']).jd
    print(time_obs)
    phot_table['time'] = time_obs
    
    results.append(phot_table)
    hdul.close()
    
#%% Combine results and save to CSV
all_results = pd.concat([r.to_pandas() for r in results], ignore_index=True)
all_results.to_csv(f'{star}_aperture_photometry_results.csv', index=False)
print(f'Aperture photometry results saved to {star}_aperture_photometry_results.csv')

#%% Example plot of the first star's aperture photometry results
plt.figure()
plt.scatter(all_results['time'], all_results['aperture_sum_bkgsub'])
plt.xlabel('Julian Date')
plt.ylabel('Aperture Sum (Background Subtracted)')
plt.title(f'Aperture Photometry Results for {star}')
plt.ylim(min(all_results['aperture_sum_bkgsub'])-50, 400)
plt.show()

# %%
