from astropy.io import fits
from datetime import datetime, timedelta
import logging
from tensorflow.keras.models import load_model
from collections import deque
import numpy as np
import pickle
import cv2
import os

# set the folder path for stream0
stream0_path = 'data/raw/example/pfrr'
# stream0_path = 'D:\stream0'

# load trained model
model_path = 'models/CNN_model'
# model_path = 'F:\pa_sample_models\CNN model'
model = load_model(
    os.path.join(model_path, 'model', 'CNN_0524.model'))

# load the binarized class labels
lb_path = os.path.join(model_path, "model/lb_4c.pickle")
lb = pickle.loads(open(lb_path, "rb").read())

# Predictions queue. The prediction is smoothed by
# the averarge of past "maxlen" frames
Q = deque(maxlen=20)

# np.array to cut the bourndary of the frames
elev_angle = np.load(os.path.join(model_path, "T_angle.npy"))
angle = 15

# get the dates available between start_date and end_date in folder_path that points to stream0 folder
def get_subfolders_in_range(start_date, end_date, folder_path=stream0_path, wavelength='558'):
    subfolder_paths = []
    current_date = start_date
    while current_date <= end_date:
        day_dir = f"{current_date.date()}-{wavelength}"
        subfolder_path = os.path.join(folder_path, day_dir)
        if os.path.exists(subfolder_path):
            subfolder_paths.append(subfolder_path)
        current_date += timedelta(days=1)
    return subfolder_paths


# helper function that decompress one folder
def decompress_pgm_files_to_dict(file_names, img_dict, num_workers=1):
    logging.info('Reading in images.')

    # Loop through each image and read into dictionary
    for file_name in file_names:

        try:
            fits_file = fits.open(file_name)
            image = fits_file[0].data.astype('uint16')
            fits_file.close()
            dt = datetime(int(file_name[-24:-20]), int(file_name[-20:-18]), int(file_name[-18:-16]),
                          int(file_name[-15:-13]), int(file_name[-13:-11]), int(file_name[-11:-9]))

            # '20200104000206'
            dt = dt.strftime('%Y%m%d%H%M%S')
            key = 'pfrr'+dt
            img_dict[key] = image

        except Exception as e:
            logging.warning(f'Issue reading in image: {e}.')
            continue

    return


def process_image_clahe(item):
    key, value = item
    dt = datetime.strptime(key[4:], '%Y%m%d%H%M%S')
    year, month, day = str(dt.year), str(dt.month), str(dt.day)
    directory_path = os.path.join(year, month, day)
    directory_path = 'data/processed/ml-classifications/pfrr/' + directory_path
    ymd_str = dt.strftime('%Y%m%d')
    time_str = dt.strftime('%H:%M:%S')

    try: 
        # process the image using clahe
        clahe = cv2.createCLAHE(clipLimit=300, tileGridSize=(8, 8))
        image = cv2.convertScaleAbs(clahe.apply(value), alpha=(255.0/65535.0))
        frame = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) # convert the frame to RGB color
        frame = cv2.resize(frame, (256, 256)).astype("float32") # resize the frame to 256 by 256 to cut the boundary
        frame[elev_angle < angle] = 0 #cut the boundary
        frame = cv2.resize(frame, (224, 224)).astype("float32") # resize the frame to 224 by 224 for prediction
        return frame, directory_path, ymd_str, time_str
    except Exception as e:
        logging.critical(f'Issue processing image: {e}.')
        return
