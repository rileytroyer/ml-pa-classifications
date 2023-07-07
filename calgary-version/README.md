# Read me for machine learning classification of pulsating aurora

Authors: Riley Troyer (science@rileytroyer.com) and Yang (Garry) Gao

## Setup

We have tested this code on a minimal Rocky 8.7 Minimal VM. 

We create a virtual python environment with miniconda3 and install the dependencies using `pip3 install -r requirements.txt`

We've also found that we need to install the additional package: `sudo yum install mesa-libGL`

As of the writing of this there is a bug in the themis_imager_readfile.read() function when using multiprocessing (when workers >1). The multiprocessing pool doesn't get closed properly, which can cause a memory leak. To fix this you can edit the source code located at: miniconda3/{venv name}/lib/pythonx.xx/site-packages/themis_imager_readfile/_themis.py

Add `pool.close()` then `pool.join()` right after the `pool.map()` function call. 

### System requirements

- Rocky 8.7 or 8.8
  
  - Likely works on other Linux systems, but we haven't tested.

- 1 or more CPU cores per VM
  
  - So far in our testing we haven't seen much of a speed up, if any, with multiple cpu cores. It is probably faster to run multiple parallel VMs each with a single core than a few with many cores.

- 7GB or more RAM per VM
  
  - The code does appear to work with a bit less than this if there is swap available, but this will cause it to run slower. 

## Running the code

The main program is within all_tasks.py. Run this script via: `python3 all_tasks.py 'YYYY-MM-DD' 'YYYY-MM-DD' num_workers`

Example: `python3 all_tasks.py '2019-01-01' '2020-12-30' 4`

This example will classify all the images within the stream0 folder between 2019-01-01 and 2020-12-30 and use 4 processes when needed. 

NOTE: the number of workers argument is optional. If none is given the script will set workers to the number of cpu cores. 

### Logging

The code logs various information as it is doing the classifications. These are written to the all_tasks.log file.

### Data

The code all_tasks.py classifies images within the stream0 folder. You can set the path to stream0 within the all_tasks_func.py file. It is currently set as `/stream0`. 

### Model

The machine learning model files are located in the CNN_model directory. The path to this can also be set within all_tasks_func.py. It is currently set as `./CNN_model`.

### Classification files

The output of all_tasks.py is a csv text file with the following columns: date, time, prediction, prediction_str, confidence

These get created in the following directory structure `./YYYY/MM/DD/yyyymmdd_{asi 4 letter}_themis_classifications.txt`

To change where these are created you will need to add the following line under line 84 in all_tasks_func.py and under line 77 in all_tasks.py

`directory_path = '{new_directory_path}' + directory_path`