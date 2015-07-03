# -*- coding: utf-8 -*-
"""
Show how to go from a basic worm to a NormalizedWorm

i.e. NormalizedWorm.from_basicWorm_factory

We then load a pre-calculated NormalizedWorm and verify that 
they are the same:   i.e. nw == nw_calculated

"""
import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append('..')
import movement_validation as mv


def test_pre_features():
    # Load from file a normalized worm, as calculated by Schafer Lab code
    base_path = os.path.abspath(mv.user_config.EXAMPLE_DATA_PATH)
    schafer_nw_file_path = os.path.join(base_path, 
                                     "example_video_norm_worm.mat")
    nw = mv.NormalizedWorm.from_schafer_file_factory(schafer_nw_file_path)

    # Load from file some non-normalized contour data, from a file we know
    # to have been generated by the same Schafer code that created the above
    # normalized worm.  This file was generated earlier in the code though,
    # and contains more "primitive", non-normalized contour and skeleton data
    schafer_bw_file_path = os.path.join(base_path, 
                                     "example_contour_and_skeleton_info.mat")  
    bw = mv.BasicWorm.from_schafer_file_factory(schafer_bw_file_path)

    # Run our tests on the loaded data.
    _test_nw_to_bw_to_nw(nw)
    _test_bw_and_nw(bw, nw)


def _test_nw_to_bw_to_nw(nw):
    print("Testing that we can load the normalized worm, then lop off "
          "all the calculated normalized worm measurements, i.e. "
          "angles, areas, length, widths, skeleton, and then re-calculate "
          "using our pre-features code, and the recalculated normalized "
          "worm will still agree with the originally loaded one to within "
          "a tolerance threshold.")
    bw_from_nw = nw.get_BasicWorm()
    nw_calculated = mv.NormalizedWorm.from_BasicWorm_factory(bw_from_nw)
    nw == nw_calculated

def _test_bw_and_nw(bw, nw):
    print("Testing that normalized worm, calculated from the Schafer basic "
          "worm using our pre-features code, is the same as the Schafer "
          "normalized worm")
    nw_calculated = mv.NormalizedWorm.from_BasicWorm_factory(bw)
    nw == nw_calculated

        
if __name__ == '__main__':
    start_time = mv.utils.timing_function()
    test_pre_features()
    print("Time elapsed: %.2fs" % 
          (mv.utils.timing_function() - start_time))
    
