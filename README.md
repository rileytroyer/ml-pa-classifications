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

## 1. Data
If you are running this for the first time you will want to download some data.

### Classification data
To download raw THEMIS images to classify run the script located at: src/data/stream0_download_final.py
This will mirror THEMIS images located at: https://data.phys.ucalgary.ca/sort_by_project/THEMIS/asi/stream0/ based on the dates and imagers in the file: data/processed/classifications/classification-03022023.xlsx

These images will be downloaded to the directory data/raw/testing/stream0

To download THEMIS data you will need to have rsync installed on your system. As this isn't available in Windows. If you are on a Windows system you will either need to run this from the virtual linux environment (WSL) or create a virtual machine.

For the PFRR images you will need to have wget installed.

### Training data
The training data needs to be human classified first, so we preprocess it and put it into an h5 file format that makes transporting it and creating videos easier.

To download all of the THEMIS training data run: src/data/download-themis-training-data.py

To downloadd all of the PFRR training data run: src/data/download-pfrr-training-data.py
