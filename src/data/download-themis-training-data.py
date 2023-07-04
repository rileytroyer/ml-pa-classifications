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
import pandas as pd
import logging

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

classification_dir = 'data/processed/classifications/'

b20220901 = classification_dir + 'classification-09092022 updated.xlsx'
b20221017 = classification_dir + 'classification-10012022.xlsx'
b20221120 = classification_dir + 'classification-11202022.xlsx'
b20230201 = classification_dir + 'classification-02012023.xlsx'

strfiles = [b20220901, b20221017, b20221120, b20230201]

days_list = []
asis_list = []

# Read in classification csvs and combine into one full dataframe
for strfile in strfiles:

    classification_csv = pd.read_excel(strfile, header=0, skiprows=0, sheet_name='themis')
    days_list.extend(list(classification_csv['date'].astype(str)))
    asis_list.extend(list(classification_csv['camera']))

# Create a new column to combine station and date
days_list = np.array(days_list)
asis_list = np.array(asis_list)

last_date = datetime(1000, 1, 1).date()
last_asi = '0000'

for n in range(0, len(days_list)):

    date_str = days_list[n]
    asi = asis_list[n]

    date = datetime(int(date[0:4]), int(date[4:6]), int(date[6:8]))

    if (date == last_date) & (asi == last_asi):
        continue

    

# Loop through each day, download, and create .h5 file
logging.info('Starting download and processing for all days.')

for day in days_list[0:1]:
    
    try: 
        # Download the images to the raw directory
        # You can use significantly more processes than cpu cores to speed this up
        download_pfrr_images(day, save_dir=data_dir+'raw/pfrr-asi/',
                            wavelength='558', processes=25)
    except Exception as e:
        logging.critical(f'Unable to download files for {day}. Stopped with error {e}')
    
    try:
        # Create the hdf5 file
        # Using more processes than cpu cores will slow this down
        pfrr_asi_to_hdf5_8bit_clahe(day, save_base_dir=data_dir+'interim/pfrr-asi-h5/',
                                    img_base_dir=data_dir+'raw/pfrr-asi/',
                                    wavelength='558', del_files=False, processes=4)
    except Exception as e:
        logging.critical(f'Unable to create h5 file for {day}. Stopped with error {e}')

logging.info('Finished downloading and processing all days.')