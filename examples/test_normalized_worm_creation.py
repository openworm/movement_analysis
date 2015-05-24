# -*- coding: utf-8 -*-
"""
Show how to go from BasicWorm to NormalizedWorm

"""

import sys, os

sys.path.append('..') 
import movement_validation

from movement_validation import utils
#from movement_validation import pre_features

user_config = movement_validation.user_config
NormalizedWorm = movement_validation.NormalizedWorm
VideoInfo = movement_validation.VideoInfo
WormFeatures = movement_validation.WormFeatures
BasicWorm = movement_validation.BasicWorm


def main():
    # Load from file a normalized worm, as calculated by Schafer Lab code
    base_path = os.path.abspath(user_config.EXAMPLE_DATA_PATH)
    schafer_file_path = os.path.join(base_path, 
                                     "example_video_norm_worm.mat")
    nw = NormalizedWorm.from_schafer_file_factory(schafer_file_path)

    # Load from file some non-normalized contour data, from a file we know
    # to have been generated by the same Schafer code that created the above
    # normalized worm.  This file was generated earlier in the code though,
    # and contains more "primitive", non-normalized contour and skeleton data
    bw_file_path = os.path.join(base_path, 
                                "example_contour_and_skeleton_info.mat")  

    bw = BasicWorm.from_schafer_file_factory(bw_file_path)
    
    nw_calculated = NormalizedWorm.from_BasicWorm_factory(bw)

    # Compare our generated normalized worm `nw2` with the pre-loaded 
    # Schafer Lab normalized worm, `nw`.  Validate they are the same.
    nw == nw_calculated

    # Jim: not sure what this is all about.  Can you document it?
    # - Michael
    other_stuff_jim(nw)


def other_stuff_jim(nw):
    # Now the goal is to go from the example_input_data to the normalized
    # worm data.
    min_worm = pre_features.MinimalWormSpecification()

    # The frame rate is somewhere in the video info. Ideally this would 
    # all come from the video parser eventually

    fps = 25.8398
    fpo = movement_validation.FeatureProcessingOptions(fps)
    video_info = VideoInfo('Example Video File', fps)

    # Generate the OpenWorm movement validation repo version of the features
    fpo.disable_feature_sections(['morphology']) 
    wf = WormFeatures(nw, video_info, fpo)
    
    wf.timer.summarize()


if __name__ == '__main__':
    main()
