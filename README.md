# Read Me File for the machine learning classification of pulsating aurora project
see document-struture.md for a layout of the project structure and where files are located
you will need to create reports/figures/, logs/, and data/ directories as these are included in my .gitignore file.

NOTE: The code and documents located within calgary-version/ are specifically designed to be run on the local University of Calgary servers that host the THEMIS data. If you are running this code make sure you do it from within calgary-version.

## Setup
1. When running code make sure you are in the base directory as this will ensure that all the code runs as expected.
2. I recommend creating a new virtual environment and installing all the dependencies through pip3 with the requirements.txt file.

### System configuration
Besides installing dependencies via the requirements file. We've tested this on a Rocky 8 VM and when doing that needed to install the package: `sudo yum install mesa-libGL`. 

There is a known issue in the themis_imager_readfile python package that can cause a memory leak. This will hopefully be fixed soon, but you can also edit the source code directly which will likely be located (if you are using a conda virtual environment) somewhere like anaconda3/envs/{environment_name}/lib/python3.11/site-packages/themis_imager_readfile/_themis.py. The fix is quite simple. Just add the following lines immediately after the `pool.map()` function call: `pool.close()` and `pool.join()`. This should fix the memory leak issue. 

** update: as of 2023-07-10 this has been fixed in the newest version of themis_imager_readfile**

## 1. Data
If you are running this for the first time you will want to download some data.

### Classification data
To download raw THEMIS images to classify run the script located at: src/data/stream0_download_final.py
This will mirror THEMIS images located at: https://data.phys.ucalgary.ca/sort_by_project/THEMIS/asi/stream0/ based on the dates and imagers in the file: data/processed/classifications/classification-03022023.xlsx

These images will be downloaded to the directory data/raw/testing/stream0

To download THEMIS data you will need to have rsync installed on your system. As this isn't available in Windows. If you are on a Windows system you will either need to run this from the virtual linux environment (WSL) or create a virtual machine.

For the PFRR images you will need to have wget installed. To download PFRR images run the code located at src/data/download-pfrr-training-data.py. This will get the images stored on the UAF ftp server: http://optics.gi.alaska.edu/amisr_archive/PKR/DASC/RAW/

### Training data
The training data needs to be human classified first, so we preprocess it and put it into an h5 file format that makes transporting it and creating videos easier.

To download all of the THEMIS training data run: src/data/download-themis-training-data.py

To downloadd all of the PFRR training data run: src/data/download-pfrr-training-data.py

### Caveats
We've noticed a few things, specifically with the THEMIS data. Occasionally within a data repository for an hour there is a data file with _wide_ in the filename. We aren't entirely sure what this is, but it has the same format as the normal _full_ files, but with bad data. This can cause some missing data and jumps in the data. However they are fairly infrequent and our newer code should look for these files and exclude them. 

There are also occasional repeat data files. One compressed and one not compressed. This will cause some repeat data, but since the data is identical shouldn't change any results. Finally it appears that around once per hour the cameras skip an image (likely to do some additional processing or something like that).

### Processing
We've found that how the images get processed can dramatically change the classification results. We've found that using contrast limited adaptive histogram equalization pulls out the auroral features the best. In our training/testing we used `cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))` for THEMIS and `cv2.createCLAHE(clipLimit=300, tileGridSize=(8, 8))` for PFRR. However in the future using `(4, 4)` for  `tileGridSize` might be a bit better and reduce some noise, especially in the PFRR camera.


## Classifying using the ML Model

### THEMIS
The model was designed specifically to classify THEMIS images. Before being put into the model, raw THEMIS images are processed using CLAHE then clipped to only include the center portion of the camera. Finally they are converted to a `(224, 224, 3).astype(float32)` array type. I honestly don't know why we need to do that last step. In future iterations of the model I think this would be worth looking into. Being able to keep the original image shape and 8-bit I believe was significantly speed up classifications. 

To actually perform classifications on images within stream0 run the script all_tasks.py. Call this from the command line using `python3 all_tasks.py 'YYYY-MM-DD' 'YYYY-MM-DD' NUM_PROCESSES`, where the first date is the date to start from and the second is the date to end on. NUM_PROCESSES isn't required. This is the number of processes to use for multiprocessing. If not specified this will default to the number of CPU cores.

More cores on your system will result in a faster code run. 

The result of this code is a text file with the classification and confidence for each image. These are output to data/processed/ml-classifications/YYYY/MM/DD/

### PFRR
I've tried a little to get the classifications working on the PFRR camera, but so far haven't had much luck getting reasonable results. I think the biggest issue is that the PFRR images are a different array size than the THEMIS ones and so the downscaling may not be working as expected. My attempt at this is located in src/models/pfrr/
