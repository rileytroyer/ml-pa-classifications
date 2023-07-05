"""
Script to download and process THEMIS ASI training data.
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
from src.data.themis_data_functions import download_themis_images, themis_asi_to_hdf5_8bit_clahe

# Important directories
data_dir = 'data/'
logs_dir = 'logs/'

# Initiate logging
logging.basicConfig(filename=logs_dir + f'download-themis-training-data-{datetime.today().date()}.log',
                    encoding='utf-8',
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')

#------------------------------Initializing done------------------------------

# import files with dates to download data for
# - ML Batch 1 and Batch 2 (the data with confidence level 4 only)
# - ML Batch 2022-09-01
# - ML Batch 2022-10-17
# - ML Batch 2022-11-20-strong-pa
# - ML Batch 2023-02-01

classification_dir = 'docs/classifications/'

b20220901 = classification_dir + 'themis/classification-09092022 updated.xlsx'
b20221017 = classification_dir + 'themis/classification-10012022.xlsx'
b20221120 = classification_dir + 'themis/classification-11202022.xlsx'
b20230201 = classification_dir + 'themis/classification-02012023.xlsx'

strfiles = [b20220901, b20221017, b20221120, b20230201]

days_list = []
asis_list = []

logging.info('Reading in classification files.')

# Read in classification csvs and combine into one full dataframe
for strfile in strfiles:

    classification_csv = pd.read_excel(strfile, sheet_name='themis')

    # Check format of date
    if type(classification_csv['Date'][0]) == str:
        days = [parser.isoparse(d) for d in classification_csv['Date']]

    else:
        days = [datetime(int(d[0:4]), int(d[4:6]), int(d[6:8]))
                for d in classification_csv['Date'].astype(str)]


    # Add days to master list
    days_list.extend(days)

    # Also extend asis
    asis_list.extend(list(classification_csv['camera']))

# Create a new column to combine station and date
days_list = np.array(days_list)
asis_list = np.array(asis_list)

last_date = datetime(1000, 1, 1)
last_asi = '0000'

# Loop through each day, download, and create .h5 file
logging.info('Starting download and processing for all days.')

for n in range(0, len(days_list)):

    date = days_list[n]
    asi = asis_list[n]

    if (date.date() == last_date.date()) & (asi == last_asi):
        continue

    logging.info(f'Downloading images for {date} and {asi} to data/raw/training/themis/.')
    # Download images
    try:
        download_themis_images(date, asi, 'data/raw/training/themis/')

    except Exception as e:
        logging.critical(f'Unable to download files for {date} and {asi}. Stopped with error {e}') 

    logging.info(f'Images downloaded. Starting to process images.')
    # Process images
    try:
        themis_asi_to_hdf5_8bit_clahe(date, asi,
                                      save_dir='data/raw/training/themis/',
                                      h5_dir='data/interim/training/themis/')  
    
    except Exception as e:
        logging.critical(f'Unable to process files for {date} and {asi}. Stopped with error {e}')

    last_date = date
    last_asi = asi

    logging.info(f'Images processed. They are available at data/interim/training/themis/.')


logging.info('All images downloaded and processed.')