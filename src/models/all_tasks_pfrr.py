#from video_generator import *
from all_tasks_func_pfrr import *
from datetime import datetime
import sys
import logging
import numpy as np
import os
import pandas as pd
from multiprocessing import Pool, cpu_count, get_context
import multiprocessing as mp
import gc

# get args from command line
if len(sys.argv) > 1:
    args = sys.argv

# set GPU devices to empty
os.environ["CUDA_VISIBLE_DEVICES"] = ""


if __name__ == '__main__':

    # print code start running
    print(f'code running, args = {args[1:]}')

    # init log file
    logging.basicConfig(filename='logs/all_tasks_pfrr.log',
                        # encoding='utf-8',
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')

    logging.info('all_task code start ' +
                 datetime.now().strftime("%H:%M:%S"))

    # use start_date and end_date to get needed folder paths
    try:
        start_date_str, end_date_str = args[1], args[2]

        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

        # format: ['pfrr/2010-10-01-558/', ...]
        subfolder_paths = get_subfolders_in_range(
            start_date, end_date, folder_path=stream0_path)
        logging.info(
            f'getting paths from {subfolder_paths[0]} to {subfolder_paths[-1]}')
    except Exception as e:
        logging.critical(f'Start or end date not valid, Exception: {e}')
        sys.exit()

    # set the num of workers for multiprocessing later. default as the cpu_count.
    try:
        if len(args)>3:
            num_workers = int(args[3])
        else:
            num_workers = cpu_count()
    except Exception as e:
        logging.critical(f'Number of processors not valid, Exception: {e}')
        sys.exit()

    # decompress the images to a dictionary
    # address example: stream0/2011/08/08/mcgr_themis11/ut09/
    # iterate through date folders
    for date_folder_path in subfolder_paths[0:1]:  # stream0/2011/08/08
        logging.info(
            f'Processing date_folder_path = {date_folder_path}, {datetime.now().strftime("%H:%M:%S")}')

        # Iterate over the child folders (each camera) in the outer folder
        for asi_name in ['pfrr']:  # /mcgr_themis11

            logging.info(
                f'Processing asi = {asi_name}')

            # create csv file in advance
            directory_path = date_folder_path[-14:]
            ymd_str = (datetime.strptime(directory_path[:-4], "%Y-%m-%d")).strftime('%Y%m%d')
            directory_path = 'data/processed/ml-classifications/pfrr/' + directory_path
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
            
            # Path to store output predictions to    
            txt_path = os.path.join(
                directory_path, ymd_str+'_'+asi_name+"_classifications.txt")
                
            # Write comment to file
            with open(txt_path, "w") as f:
                # create the comment section
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                comment = f"# File created on {now}\n# This file contains the predictions generated by the model.\n\n"
                f.write(comment)
                
            # init a dataframe to store information
            df = pd.DataFrame(
                columns=['date', 'time', 'prediction', 'prediction_str', 'confidence'])

            # Get list of all files for day
            image_files = os.listdir(date_folder_path)
            image_files = sorted([f for f in image_files if f.endswith('.FITS')])

            ### ! Processing starts from here
            # if empty hours list, continue
            if len(image_files) == 0:
                logging.info(
                    f'DATE SKIPPED: camera_dict empty, asi_name = {asi_name}, date = {date_folder_path}')
                continue

            # Loop through 100 images at a time
            image_batch_size = 100
            for n, image_file in enumerate(image_files[::image_batch_size]):

                batch_image_files = image_files[n*image_batch_size:(n+1)*image_batch_size]
                batch_image_files = [os.path.join(date_folder_path, f) for f in batch_image_files]
            
                logging.info(f'Reading in {n*image_batch_size} images of {len(image_files)}.')
                # camera_dict example k-v pair: {'atha20200104000206':img[:,:,:]}
                camera_dict = {}
                decompress_pgm_files_to_dict(batch_image_files, camera_dict, num_workers=num_workers)
                
                try:
                    if not camera_dict:
                        logging.info(
                            f'DATE SKIPPED: camera_dict empty, asi_name = {asi_name}, date = {date_folder_path}')
                        continue
                except Exception as e:
                    logging.CRITICAL(f'Error occurs in making dataframe as {e}')
                    logging.CRITICAL(f'HOUR SKIPPED: asi_name = {asi_name}, date = {date_folder_path}, hour = {hour}')
                    continue # if exception, go to next hour

                # try multiprocessing steps
                try: 
                    logging.info(f'Images read in. Starting processing via multiprocessing.')
                    # Create a pool of worker processes
                    num_workers = num_workers
                    pool = get_context("spawn").Pool(processes=num_workers)
                    logging.info(f'Pool generated, num_workers = {num_workers}.')

                    # Map the process_image function to each item in camera_dict using multiprocessing
                    results = pool.map(process_image_clahe, camera_dict.items())
                    
                    # Close the multiprocessing pool
                    pool.close()
                    pool.join()
                    logging.info('Pool joined.')
                    
                    # Assign output to empty lists
                    frames, directory_paths, ymd_strs, time_strs = [], [], [], []
                    
                    # Loop through each frame and append to list
                    for result in results:
                        frame, directory_path, ymd_str, time_str = result
                        frames.append(frame)
                        ymd_strs.append(ymd_str)
                        time_strs.append(time_str)
                        
                    logging.info('Images processed. Starting model predictions.')
                    
                except Exception as e:
                    logging.critical(f'Issue processing images: {e}.')
                    continue
                    
                try:  
                    # Convert model input into array
                    frames = np.array(frames)
                    preds = model.predict(frames, batch_size=30)
                    
                    # Garbage collection to deal with memory leak from model.predict
                    _ = gc.collect()
                    
                    # Write model results into dataframe
                    prediction_nums = list(map(np.argmax, preds))
                    confidences = list(map(np.max, preds))
                    prediction_strs = [lb.classes_[item] for item in prediction_nums]
                    new_rows = pd.DataFrame({'date': ymd_strs, 'time': time_strs, 'prediction': prediction_nums, 'prediction_str': prediction_strs, 'confidence': confidences}) 
                    # Close the pool of worker processes
                    logging.info(f'Model prediction finished.')
                    # Append the processed rows to the DataFrame
                    df = pd.concat([df, new_rows], ignore_index=True)
        

                except Exception as e:
                    logging.CRITICAL(f'Issue with model prediction: {e}.')
                    logging.CRITICAL(
                        f'DATE SKIPPED: asi_name = {asi_name}, date = {date_folder_path}')
                    continue  # if exception, go to next asi camera

            # Write dataframe to csv file
            df.to_csv(txt_path, mode='a', index=False, header=True)
                    
            logging.info(f'Predictions written to file and available at {txt_path}.')
        
        
            logging.info(f'date_folder_path={date_folder_path}, asi={asi_name} results generated, time = {datetime.now().strftime("%H:%M:%S")}')
