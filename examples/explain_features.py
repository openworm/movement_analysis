# -*- coding: utf-8 -*-
"""
Used in IPython notebooks. Running as a serparate script will generate plots and text to console.

To run this code files should be obtained from:
https://drive.google.com/folderview?id=0B7to9gBdZEyGNWtWUElWVzVxc0E&usp=sharing

In addition the user_config.py file should be created in the 
movement_validation package based on the user_config_example.txt

"""

import sys, os

# We must add .. to the path so that we can perform the 
# import of movement_validation while running this as 
# a top-level script (i.e. with __name__ = '__main__')
sys.path.append('..') 

print(sys.path)

from movement_validation import NormalizedWorm

from movement_validation import user_config, NormalizedWorm
from movement_validation import WormFeatures, utils


def main():
    """
    Generate plots and text explaining feature calculation process.
    Requires arguement indicating feature:
    range
    etc

    """

    explain = []

    if __debug__:
        if not len(sys.argv) > 1:
            raise AssertionError('Please indicated which feature you would like explained.')
        else:
            explain = sys.argv[1:]

    # Set up the necessary file paths for file loading
    #----------------------
    base_path = os.path.abspath(user_config.EXAMPLE_DATA_PATH)

    matlab_generated_file_path = os.path.join(
        base_path,'example_video_feature_file.mat')

    data_file_path = os.path.join(base_path,"example_video_norm_worm.mat")

    # OPENWORM
    #----------------------
    # Load the normalized worm from file
    nw = NormalizedWorm.from_schafer_file_factory(data_file_path)

    #The frame rate is somewhere in the video info. Ideally this would all come
    #from the video parser eventually
    #vi = VideoInfo('Example Video File',25.8398)

    # Generate the OpenWorm movement validation repo version of the features

    print(explain)

    openworm_features = WormFeatures(nw, explain=explain) #need to add flags to functions to enable expaination output


if __name__ == '__main__':
    main()
