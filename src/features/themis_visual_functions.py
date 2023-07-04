""" 
Functions used to turn images into images/movies for THEMIS ASI Images

@author Riley Troyer
science@rileytroyer.com
"""

from datetime import datetime
import gc
from dateutil import parser
import h5py
import logging
import math
from matplotlib import animation
from matplotlib import pyplot
from moviepy.editor import *
import multiprocessing
from multiprocessing import Pool
import numpy
import os
import re

def movie_job(job_input:list):
    """Function to create timestamped movie from input images and times. 
    Outputs to .mp4 file with specified filename.
    INPUT
    job_input - [filepathname.mp4, list of datetimes, array of images]
    OUTPUT
    None
    """
    # Get times from input
    all_times = job_input[1]

    # Get all the images too
    all_images = job_input[2]

    # CREATE MOVIE
    img_num = all_images.shape[2]
    fps = 20.0


    # Construct an animation
    # Setup the figure
    fig, axpic = pyplot.subplots(1, 1)

    # No axis for images
    axpic.axis('off')

    # Plot the image
    img = axpic.imshow(numpy.flipud(all_images[:, :, 0]),
                       cmap='gray', animated=True)

    # Add frame number and timestamp to video
    frame_num = axpic.text(10, 250, '00000', fontweight='bold',
                           color='red')
    time_str = str(all_times[0])
    time_label = axpic.text(120, 250,
                            time_str,
                            fontweight='bold',
                            color='red')

    pyplot.tight_layout()

    def updatefig(frame:numpy.ndarray) -> "[numpy.ndarray, int, str]":
        """Function to update the animation"""

        # Set new image data
        img.set_data(numpy.flipud(all_images[:, :, frame]))
        # And the frame number
        start_frame = int(job_input[0].split('/')[-1].split('.')[0][11:])
        frame_num.set_text(str(frame + start_frame).zfill(5))
        #...and time
        time_str = str(all_times[frame])
        time_label.set_text(time_str)

        return [img, frame_num, time_label]

    # Construct the animation
    anim = animation.FuncAnimation(fig, updatefig,
                                   frames=img_num,
                                   interval=int(1000.0/fps),
                                   blit=True)

    # Close the figure
    pyplot.close(fig)


    # Use ffmpeg writer to save animation
    event_movie_fn = (job_input[0])
    writer = animation.writers['ffmpeg'](fps=fps)
    anim.save(event_movie_fn,
              writer=writer, dpi=150)

def create_timestamped_movie(date:datetime, asi:str, save_dir:str, workers:int=1):
    """Function to create a movie from THEMIS ASI files with a timestamp and frame number.
    Includes a timestamp, and frame number. 
    INPUT
    date - day to process image files for
    asi - 4 letter themis asi location
    save_dir - where h5 file images are stored.
    workers - how many processes to create movies with.
    OUTPUT
    none
    """

    # Select file with images
    logging.info('Starting timestamped movie script for {} and {}.'.format(asi,
                                                                           date.date()))

    img_file = (save_dir + asi + '/all-images-'
                + str(date.date()) + '-' + asi + '.h5')

    logging.info('Reading in h5 file: {}'.format(img_file))

    try:
        # Read in h5 file
        themis_file = h5py.File(img_file, "r")

        # Get times from file
        all_times = [parser.isoparse(d) for d in pfrr_file['iso_ut_time']]

        # Get all the images
        all_images = themis_file['images']

    except Exception as e:
        logging.critical('There was an issue reading in the h5 file. Stopping.')
        logging.critical('Exception: {}'.format(e))
        raise

    # Check if directory to store movies exists
    movie_dir = save_dir + asi + '/movies/'
    if not os.path.exists(movie_dir):
        os.makedirs(movie_dir)

    # Check if directory to store temporary frames exists
    if not os.path.exists(save_dir + 'tmp-frames/'):
        os.makedirs(save_dir + 'tmp-frames/')


    # Split images and times into smaller portions
    # this allows us to speed up the process with parallel computing

    # How many smaller movies to make, these will get combined into 1 at the end
    bins = 10
    chunk_size = math.ceil(len(all_times)/bins)

    # Define a list to be able to input into the parallelization job
    job_input = []

    # Loop through each chunk and set as one job input
    for n in range(0, len(all_times), chunk_size):

        # Filename for movie chunk
        filename = save_dir + 'tmp-frames/tmp-frames-' + str(n) + '.mp4'

        # Append to job input, need filename, times, and images
        job_input.append([filename, all_times[n:n+chunk_size],
                          all_images[:, :, n:n+chunk_size]])

    # Delete first image array to save ram as this is pretty big
    del all_images, all_times
    gc.collect()

    # Start multiprocessing
    # be aware this can use a fairly large amount of RAM.
    # I often see around 5GB used.
    logging.info('Starting {} movie creating processes.'
                 ' Tmp movies will be combined into one at the end.'.format(workers))

    try:
        with multiprocessing.get_context("forkserver").Pool(processes=workers) as pool:
            pool.map(movie_job, job_input)

        # Terminate threads when finished
        pool.terminate()
        pool.join()

    except Exception as e:
        logging.critical('There was an issue creating the tmp movie files. Stopping.')
        logging.critical('Exception: {}'.format(e))
        raise

    logging.info('Finished creating tmp movies.')

    # List of all tmp movies
    tmp_movie_files = [f for f in os.listdir(save_dir + 'tmp-frames/')
                       if f.startswith('tmp') & f.endswith('.mp4')]

    # Make sure files are sorted properly
    def num_sort(string):
        return list(map(int, re.findall(r'\d+', string)))[0]

    tmp_movie_files.sort(key=num_sort)

    # Add in path
    tmp_movie_files = [save_dir + 'tmp-frames/' + f for f in tmp_movie_files]

    # File to write
    full_movie_pathname = movie_dir + 'full-movie-' + str(date.date()) + '-' + asi + '.mp4'

    # Concatenate smaller tmp movies into a full one
    logging.info('Combining tmp movies into one file at: {}.'.format(full_movie_pathname))
    try:
        clips = []

        for filename in tmp_movie_files:
            clips.append(VideoFileClip(filename))

        video = concatenate_videoclips(clips, method='chain')
        video.write_videofile(full_movie_pathname)

    except Exception as e:
        logging.warning('There was an issue creating the full movie file. Stopping.')
        logging.warning('Exception: {}'.format(e))

    logging.info('Full movie file created. Deleting tmp files.')

    try:
        # Remove all tmp movie files
        for file in tmp_movie_files:
            os.remove(file)

    except Exception as e:
        logging.warning('Could not delete tmp movie files.')
        logging.warning('Exception: {}'.format(e))

    logging.info('Finished timestamped movie script for {} and {}.'.format(asi,
                                                                           date.date()))
