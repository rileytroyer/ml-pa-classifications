""" 
Functions used to download and process THEMIS ASI Images

@author Riley Troyer
science@rileytroyer.com
"""

import cv2
from datetime import datetime
import gc
import h5py
import logging
import numpy
import os
from scipy.io import readsav
import shutil
import subprocess
import themis_imager_readfile


def download_themis_images(date:datetime, asi:str, save_dir:str):
    """Function to download raw .pgm.gz files from stream0 of THEMIS
    ASIs. Data is downloaded from https://data.phys.ucalgary.ca/
    INPUT
    date - which date to download images for
    asi - 4 letter code for station to download images for
    save_dir - where to save images files
    OUTPUT
    logging. I recommend writing to file by running this at the start of the code:
    
    logging.basicConfig(filename='themis-script.log',
                    encoding='utf-8',
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')
    """
    
    logging.info('Starting download script for {} and {}.'.format(asi, date.date()))
    
    date_string = (str(date.year).zfill(4) + '/' 
                   + str(date.month).zfill(2) + '/'
                   + str(date.day).zfill(2) + '/')

    # URL for entire THEMIS ASI project
    themis_url = 'data.phys.ucalgary.ca/data/sort_by_project/THEMIS/asi/'

    # URL for skymap
    skymap_url = themis_url + 'skymaps/' + asi + '/'

    # URL for images
    img_url = themis_url + 'stream0/' + date_string + asi + '*/'

    # Get the matched skymap for the date

    # Get output of sync to find the directories
    try:
        skymap_dirs = subprocess.check_output(['rsync', 
                                               'rsync://' + skymap_url]).splitlines()
        skymap_dirs = [str(d.split(b' ')[-1], 'UTF-8') for d in skymap_dirs[1:]]

    except Exception as e:
        logging.critical('Unable to access skymap server: {}. '
                         'Server may be down. Stopping.'.format(skymap_url))
        logging.critical('Exception: {}'.format(e))
        raise

    # Convert to datetimes
    skymap_dates = [d.split('_')[1] for d in skymap_dirs]
    skymap_dates = [datetime.strptime(d, '%Y%m%d') for d in skymap_dates]

    # Find time difference from each skymap date
    time_diffs = numpy.array([(date - d).total_seconds() for d in skymap_dates])

    # Find the closest map
    skymap_dir = skymap_dirs[numpy.where(time_diffs > 0,
                                         time_diffs, numpy.inf).argmin()]

    skymap_url = skymap_url + skymap_dir + '/'

    # Create directories to store data

    # Does directory exist for imager?
    save_asi_dir = save_dir + asi + '/'
    if not os.path.exists(save_asi_dir):
        os.makedirs(save_asi_dir)

    # Does a temporary directory for raw images and skymap files exist?
    tmp_dir = save_asi_dir + 'tmp/'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    tmp_img_dir = tmp_dir + str(date.date()) + '/'
    if not os.path.exists(tmp_img_dir):
        os.makedirs(tmp_img_dir)
        download_imgs = True
    else:
        logging.info('Images already exists for {},'
                     ' so will not download.'.format(date.date()))
        download_imgs = False

    # Do the images need to be downloaded?
    if download_imgs == True:
        
        # Download skymap
        logging.info('Downloading skymap from {}...'.format(skymap_url))
        try:
            subprocess.run(['rsync', '-vzrt', 'rsync://' + skymap_url + '*.sav',
                             tmp_img_dir], stdout=subprocess.DEVNULL)
            logging.info('Successfully downloaded skymap.'
                             ' It is saved at {}.'.format(tmp_img_dir))
        except Exception as e:
            logging.critical('Unable to download skymap:{}. Stopping.'.format(skymap_url))
            logging.critical('Exception: {}'.format(e))
            raise

        # Download images
        logging.info('Downloading images from {}...'.format(img_url))
        try:
            subprocess.run(['rsync', '-vzrt', 'rsync://' + img_url,
                             tmp_img_dir], stdout=subprocess.DEVNULL)
            logging.info('Successfully downloaded images.'
                         ' They are saved at {}.'.format(tmp_img_dir))
        except Exception as e:
            logging.critical('Unable to download images:{}. Stopping.'.format(img_url))
            logging.critical('Exception: {}'.format(e))
            raise
            
    logging.info('Finished download script for {} and {}.'.format(asi, date.date()))


def themis_asi_to_hdf5_8bit_clahe(date:datetime, asi:str, save_dir, del_files:bool = False,
                                  workers:int=1):
    """Function to convert themis asi images
    to 8-bit grayscale images and then write them to an h5 file using
    contrast limited adaptive historgram equalization (CLAHE).
    INPUT
    date - date to perform image conversion and storage for
    asi - which THEMIS camera to use
    save_dir - where are images stored
    del_files - whether to delete the individual files after program runs
    workers - how many multiprocessing workers to use when reading in raw themis files
              be aware there is a memory leak issue in the themis_imager_readfile code
              as of the writting of this. When calling this function many times with multiprocessing
              the code doesn't close out the multiprocessing pool correctly causing the issue.
    OUTPUT
    logging. I recommend writing to file by running this at the start of the code:
    
    logging.basicConfig(filename='themis-script.log',
                    encoding='utf-8',
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')
    """

    def process_image(image:numpy.ndarray) -> numpy.ndarray:
        """CLAHE processing of 16-bit image and downscale to 8-bit"""
        # process the image using clahe
        clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))
        image = clahe.apply(image)
 
        return cv2.convertScaleAbs(image, alpha=(255.0/65536.0))
    
    # Write images to h5 dataset
    logging.info('Starting h5 file creation script for {} and {}...'.format(asi,
                                                                            date.date()))

    h5file = save_dir + asi + '/all-images-' + str(date.date()) + '-' + asi + '.h5'
    
    # Directory with images
    tmp_img_dir = save_dir + asi + '/tmp/' + str(date.date()) + '/'
    
    if not os.path.exists(tmp_img_dir):
        logging.critical('Images are not downloaded. Try running download_themis_images.')
    
    # Read in skymap
    skymap_file = [f for f in os.listdir(tmp_img_dir) if f.endswith('.sav')][0]

    try:
        # Try reading IDL save file
        skymap = readsav(tmp_img_dir + skymap_file, python_dict=True)['skymap']

        # Get arrays
        skymap_alt = skymap['FULL_MAP_ALTITUDE'][0]
        skymap_glat = skymap['FULL_MAP_LATITUDE'][0][:, 0:-1, 0:-1]
        skymap_glon = skymap['FULL_MAP_LONGITUDE'][0][:, 0:-1, 0:-1]
        skymap_elev = skymap['FULL_ELEVATION'][0]
        skymap_azim = skymap['FULL_AZIMUTH'][0]
        
        logging.info('Read in skymap file from: {}'.format(skymap_file))
        
    except Exception as e:
        logging.error('Unable to read skymap file: {}.'
                         ' Creating file without it.'.format(tmp_img_dir + skymap_file))
        logging.error('Exception: {}'.format(e))
        
        skymap_alt = numpy.array(['Unavailable'])
        skymap_glat = numpy.array(['Unavailable'])
        skymap_glon = numpy.array(['Unavailable'])
        skymap_elev = numpy.array(['Unavailable'])
        skymap_azim = numpy.array(['Unavailable'])

    # Does the downloaded image directory exists?
    if not os.path.exists(tmp_img_dir):
        logging.critical('Images do not exist at {}'.format(tmp_img_dir))

    hour_dirs = os.listdir(tmp_img_dir)
    hour_dirs = sorted([d for d in hour_dirs if d.startswith('ut')])

    # Construct a list of pathnames to each file for day
    filepathnames = []

    for hour_dir in hour_dirs:

        # Name of all images in hour
        img_files = sorted(os.listdir(tmp_img_dir + hour_dir))
        img_files = [tmp_img_dir + hour_dir + '/' + f for f in img_files]

        # Add to master list
        filepathnames.append(img_files)

    with h5py.File(h5file, 'w') as h5f:

        # Initialize the datasets for images and timestamps
        img_ds = h5f.create_dataset('images', shape=(256, 256, 0),
                                    maxshape=(256, 256, None),
                                    dtype='uint8')

        time_ds = h5f.create_dataset('iso_ut_time', shape=(0,),
                                     maxshape=(None,),
                                     dtype='S27')

        alt_ds = h5f.create_dataset('skymap_alt', shape=skymap_alt.shape,
                                     dtype='float', data=skymap_alt)        
        
        glat_ds = h5f.create_dataset('skymap_glat', shape=skymap_glat.shape,
                                     dtype='float', data=skymap_glat)

        glon_ds = h5f.create_dataset('skymap_glon', shape=skymap_glon.shape,
                                     dtype='float', data=skymap_glon)

        elev_ds = h5f.create_dataset('skymap_elev', shape=skymap_elev.shape,
                                     dtype='float', data=skymap_elev)

        azim_ds = h5f.create_dataset('skymap_azim', shape=skymap_azim.shape,
                                     dtype='float', data=skymap_azim)

        # Loop through each hour, process and write images to file
        logging.info('Processing and writing images to file...')

        # Create dictionary to store images in
        camera_dict = {}

        try:
            for hour_filepathnames in filepathnames:
                # logging.info('file name is {}'.format(hour_filepathnames))

                # Read the data files
                images, meta, problematic_files = themis_imager_readfile.read(hour_filepathnames,
                                                                              workers=workers)

                # Check if images exists
                if images.shape[2] == 0:
                    logging.warning(f'No images for hour, skipping.')
                    continue

                # Extract datetimes from file
                datetimes = [datetime.strptime(m['Image request start'],
                                                 '%Y-%m-%d %H:%M:%S.%f %Z') for m in meta]

                # Convert times to integer format
                timestamps = numpy.array([t.isoformat() + 'Z' for t in datetimes]).astype('S27')

                # Process the images
                for n in range(images.shape[2]):
                    images[:, :, n] = process_image(images[:, :, n])


                # Write image to dataset. This requires resizing
                # logging.info("img shape {}".format(img.shape))
                img_ds.resize(img_ds.shape[2] + images.shape[2], axis=2)
                img_ds[:, :, -images.shape[2]:] = images

                # Write timestamp to dataset
                time_ds.resize(time_ds.shape[0] + timestamps.shape[0], axis=0)
                time_ds[-timestamps.shape[0]:] = timestamps

        except Exception as e:
            logging.critical('Unable to write images to file. Stopping.'
                             ' Deleting h5 file and, if specified, images.')
            logging.critical('Exception: {}'.format(e))
            
            # Delete h5 file
            os.remove(h5file)
            
            # Delete the raw image files if specified
            if del_files == True:
                logging.info('Deleting directory: {}'.format(tmp_img_dir))
                shutil.rmtree(tmp_img_dir)
            raise

        # Add attributes to datasets
        time_ds.attrs['about'] = ('ISO 8601 formatted timestamp in byte string.')
        img_ds.attrs['wavelength'] = 'white'
        img_ds.attrs['station_latitude'] = float(meta[0]['Geodetic latitude'])
        img_ds.attrs['station_longitude'] = float(meta[0]['Geodetic Longitude'])
        alt_ds.attrs['about'] = 'Altitudes for different skymaps.'
        glat_ds.attrs['about'] = 'Geographic latitude at pixel corner, excluding last.'
        glon_ds.attrs['about'] = 'Geographic longitude at pixel corner, excluding last.'
        elev_ds.attrs['about'] = 'Elevation angle of pixel center.'
        azim_ds.attrs['about'] = 'Azimuthal angle of pixel center.'
        
    # Delete the raw image files if specified
    if del_files == True:
        logging.info('Deleting directory: {}'.format(tmp_img_dir))
        shutil.rmtree(tmp_img_dir)

    logging.info('Finished h5 file creation script for {} and {}.'
                 ' File is saved to: {}'.format(asi, date.date(), h5file))