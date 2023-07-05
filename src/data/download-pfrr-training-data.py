"""
Script to download and process PFRR ASI training data.
Processed data is stored in an hdf5 file (.h5) with all the images for an entire day.
Processed images are downscaled from 16-bit to 8-bit using a custom scaling.
Custom scaling uses contrast limited adaptive histogram equalization

Written by Riley Troyer
science@rileytroyer.com
"""
# Import needed libraries
from datetime import datetime
from dateutil import parser
import pandas as pd
from pathlib import Path
import logging
import numpy as np
import sys

# Add root to path
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

# Import functions to do downloading and processing
from src.data.pfrr_data_functions import download_pfrr_images, pfrr_asi_to_hdf5_8bit_clahe

# Important directories
data_dir = 'data/'
logs_dir = 'logs/'

# Initiate logging
logging.basicConfig(filename=logs_dir + f'download-pfrr-training-data-{datetime.today().date()}.log',
                    encoding='utf-8',
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')

#------------------------------Initializing done------------------------------

# import files with dates to download data for

classification_dir = 'docs/classifications/'

b1 = classification_dir + 'pfrr/New ML Batch 1 updated.xlsx'
b2 = classification_dir + 'pfrr/New ML Batch 2 updated.xlsx'

strfiles = [b1, b2]

days_list = []

logging.info('Reading in classification files.')

# Read in classification csvs and combine into one full dataframe
for strfile in strfiles:

    classification_csv = pd.read_excel(strfile, sheet_name='pfrr')

    days = [d.to_pydatetime() for d in classification_csv['Date']]

    # Add days to master list
    days_list.extend(days)

# Create a new column to combine station and date
days_list = np.unique(days_list)

# Loop through each day, download, and create .h5 file
logging.info('Starting download and processing for all days.')

for date in days_list:
    
    logging.info(f'Downloading images for {date} to data/raw/training/pfrr/.')
    try: 
        # Download the images to the raw directory
        # You can use significantly more processes than cpu cores to speed this up
        download_pfrr_images(date.date(), save_dir='data/raw/training/pfrr/',
                            wavelength='558', processes=25)
    except Exception as e:
        logging.critical(f'Unable to download files for {date}. Stopped with error {e}')
    
    logging.info(f'Images downloaded. Starting to process images.')
    try:
        # Create the hdf5 file
        # Using more processes than cpu cores will slow this down
        pfrr_asi_to_hdf5_8bit_clahe(date.date(), save_dir='data/raw/training/pfrr/',
                                    h5_dir='data/interim/training/pfrr/',
                                    wavelength='558', del_files=False, processes=4)
    except Exception as e:
        logging.critical(f'Unable to create h5 file for {date}. Stopped with error {e}')

    logging.info(f'Images processed. They are available at data/interim/training/pfrr/.')

logging.info('Finished downloading and processing all days.')