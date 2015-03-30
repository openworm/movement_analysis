# -*- coding: utf-8 -*-
"""
This module defines the NormalizedWorm class

"""

import numpy as np
import scipy.io

import warnings
import os
import inspect
import h5py

from scipy.signal import savgol_filter as sgolay
#Why didn't this work?
#import scipy.signal.savgol_filter as sgolay
#http://stackoverflow.com/questions/29324814/what-are-the-rules-for-importing-with-as-in-python-without-using-from

from . import config
from . import utils

import time

class NormalizedWorm(object):
    """ 
    NormalizedWorm encapsulates the normalized measures data, loaded
    from the two files, one for the eigenworm data and the other for 
    the rest.

    This will be an intermediate representation, between the parsed,
    normalized worms, and the "feature" sets. The goal is to take in the
    code from normWorms and to have a well described set of properties
    for rewriting the feature code.

    Attributes:
    -----------
      segmentation_status   
      frame_codes
      X vulva_contours        49 x 2 x n_frames
      X non_vulva_contours    49 x 2 x n_frames
      X skeletons : numpy.array
          - (49,2,n_frames)
      X angles : numpy.array
          - (49,n_frames)
      in_out_touches ????? 49 x n_frames
      lengths : numpy.array
          - (nframes,)
      widths : numpy.array
          - (49,n_frames)
      head_areas :
          - (n_frames,)
      tail_areas :
          - (n_frames,)
      vulva_areas :
          - (n_frames,)
      non_vulva_areas :
          - (n_frames,)

    Derived Attributes:
    -------------------
      n_frames
      x - how does this differ from skeleton_x???
      y
      contour_x
      contour_y
      skeleton_x

    static methods:
      getObject              load_normalized_data(self, data_path)

    """
    
    # The normalized worm contains precisely 49 points per frame.  Here
    # we list in a dictionary various partitions of the worm.
    # These are RANGE values, so the last value is not inclusive
    worm_partitions = {'head': (0, 8),
                       'neck': (8, 16),
                       'midbody':  (16, 33),
                       'old_midbody_velocity': (20, 29),
                       'hips':  (33, 41),
                       'tail': (41, 49),
                       # refinements of ['head']
                       'head_tip': (0, 4),
                       'head_base': (4, 8),    # ""
                       # refinements of ['tail']
                       'tail_base': (40, 45),
                       'tail_tip': (45, 49),   # ""
                       'all': (0, 49),
                       # neck, midbody, and hips
                       'body': (8, 41)}

    # this stores a dictionary of various ways of organizing the partitions
    worm_partition_subsets = {'normal': ('head', 'neck', 'midbody', 'hips', 'tail'),
                             'first_third': ('head', 'neck'),
                             'second_third': ('midbody',),
                             'last_third': ('hips', 'tail'),
                             'all': ('all',)}    

    N_POINTS_NORMALIZED = 49

    def __init__(self,skeletons,vulva_contours,non_vulva_contours,is_valid):
        """ 
        Create an instance from skeleton and contour data
        
        Parameters
        --------------------------------------- 
        skeletons : list
            Each element in the list may be empty (bad frame) or be of size
            2 x n, where n varies depending on the # of pixels the worm occupied
            in the given frame
            
        vulva_contours and non_vulva_contours should start and end at the same locations, from head to tail

        """

        self.angles = WormParsing.calculateAngles
        
        #TODO: Do I want to grab the skeletons from here????
        widths = WormParsing.computeWidths(self,vulva_contours,non_vulva_contours)            
        
        #t = time.time()
        self.skeletons = WormParsing.normalizeAllFramesXY(self,skeletons)   
        self.vulva_contours = WormParsing.normalizeAllFramesXY(self,vulva_contours)
        self.non_vulva_contours = WormParsing.normalizeAllFramesXY(self,non_vulva_contours)   
        #elapsed = time.time() - t
        
        self.lengths = WormParsing.computeSkeletonLengths(self,skeletons)
  
        import pdb  
        pdb.set_trace()  
        
    
        
        
        """
        From Ev's Thesis:
        3.3.1.6 - page 126 (or 110 as labeled in document)
        For each section, we begin at its center on both sides of the contour. We then
        walk, pixel by pixel, in either direction until we hit the end of the section on
        opposite sides, for both directions. The midpoint, between each opposing pixel
        pair, is considered the skeleton and the distance between these pixel pairs is
        considered the width for each skeleton point.
        3) Food tracks, noise, and other disturbances can form spikes on the worm
        contour. When no spikes are present, our walk attempts to minimize the width
        between opposing pairs of pixels. When a spike is present, this strategy may
        cause one side to get stuck in the spike while the opposing side walks.
        Therefore, when a spike is present, the spiked side walks while the other side
        remains still.
        """
        
        #Areas:
        #------------------------------------
            
        """
        Final needed attributes:
        ------------------------
        #1) ??? segmentation_status   
        #2) ??? frame_codes
        #3) DONE vulva_contours        49 x 2 x n_frames
        #4) DONE non_vulva_contours    49 x 2 x n_frames
        #5) DONE skeletons : numpy.array
          - (49,2,n_frames)
        #6) DONE angles : numpy.array
          - (49,n_frames)
        #7) in_out_touches ????? 49 x n_frames
        #8) DONE lengths : numpy.array
          - (nframes,)
        #9) widths : numpy.array
          - (49,n_frames)
        #10) head_areas
        #11) tail_areas
        #12) vulva_areas
        #13) non_vulva_areas 
        
        
        """
            
            
        
        import pdb
        pdb.set_trace()
    

    
    


    @classmethod
    def load_matlab_data(cls, data_file_path):
        """ 
        Load the norm_obj.mat file into this class

        Notes
        ---------------------------------------    
        Translated from getObject in SegwormMatlabClasses

        """
        
        self = cls.__new__(cls)         
        
        # DEBUG: (Note from @MichaelCurrie:)
        # This should be set by the normalized worm file, since each
        # worm subjected to an experiment is manually examined to find the
        # vulva so the ventral mode can be determined.  Here we just set
        # the ventral mode to a default value as a stopgap measure
        self.ventral_mode = config.DEFAULT_VENTRAL_MODE          
        

        if(not os.path.isfile(data_file_path)):
            raise Exception("Data file not found: " + data_file_path)
        else:
            self.data_file = scipy.io.loadmat(data_file_path,
                                              # squeeze unit matrix dimensions:
                                              squeeze_me=True,
                                              # force return numpy object
                                              # array:
                                              struct_as_record=False)

            # self.data_file is a dictionary, with keys:
            # self.data_file.keys() =
            # dict_keys(['__header__', 's', '__version__', '__globals__'])

            # All the action is in data_file['s'], which is a numpy.ndarray where
            # data_file['s'].dtype is an array showing how the data is structured.
            # it is structured in precisely the order specified in data_keys
            # below

            staging_data = self.data_file['s']

            # NOTE: These are aligned to the order in the files.
            # these will be the keys of the dictionary data_dict
            data_keys = [
                # a string of length n, showing, for each frame of the video:
                # s = segmented
                # f = segmentation failed
                # m = stage movement
                # d = dropped frame
                # n??? - there is reference tin some old code to this
                # after loading this we convert it to a numpy array.
                'segmentation_status',
                # shape is (1 n), see comments in
                # seg_worm.parsing.frame_errors
                'frame_codes',
                'vulva_contours',     # shape is (49, 2, n) integer
                'non_vulva_contours',  # shape is (49, 2, n) integer
                'skeletons',          # shape is (49, 2, n) integer
                'angles',             # shape is (49, n) integer (degrees)
                'in_out_touches',     # shpe is (49, n)
                'lengths',            # shape is (n) integer
                'widths',             # shape is (49, n) integer
                'head_areas',         # shape is (n) integer
                'tail_areas',         # shape is (n) integer
                'vulva_areas',        # shape is (n) integer
                'non_vulva_areas',    # shape is (n) integer
                'x',                  # shape is (49, n) integer
                'y']                  # shape is (49, n) integer

            
            for key in data_keys:
                setattr(self, key, getattr(staging_data, key))
            
            # Let's change the string of length n to a numpy array of single
            # characters of length n, to be consistent with the other data
            # structures
            self.segmentation_status = np.array(list(self.segmentation_status))

            self.load_frame_code_descriptions()

        return self

    def load_frame_code_descriptions(self):
        """
        Load the frame_codes descriptions, which are stored in a .csv file

        """
        # Here we assume the CSV is located in the same directory 
        # as this current module's directory.
        file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 'frame_codes.csv')
        f = open(file_path, 'r')

        self.frame_codes_descriptions = []

        for line in f:
            # split along ';' but ignore any newlines or quotes
            a = line.replace("\n", "").replace("'", "").split(';')
            # the actual frame codes (the first entry on each line)
            # can be treated as integers
            a[0] = int(a[0])
            self.frame_codes_descriptions.append(a)

        f.close()


    def get_partition_subset(self, partition_type):
        """ 
        There are various ways of partitioning the worm's 49 points.
        this method returns a subset of the worm partition dictionary

        TODO: This method still is not obvious to me. Also, we should move
        these things to a separate class.

        Parameters
        ---------------------------------------
        partition_type: string
          e.g. 'head'

        Usage
        ---------------------------------------
        For example, to see the mean of the head and the mean of the neck, 
        use the partition subset, 'first_third', like this:

        nw = NormalizedWorm(....)

        width_dict = {k: np.mean(nw.get_partition(k), 0) for k in ('head', 'neck')}

        OR, using self.worm_partition_subsets,

        s = nw.get_paritition_subset('first_third')
        # i.e. s = {'head':(0,8), 'neck':(8,16)}

        width_dict = {k: np.mean(nw.get_partition(k), 0) for k in s.keys()}

        Notes
        ---------------------------------------    
        Translated from get.ALL_NORMAL_INDICES in SegwormMatlabClasses / 
        +seg_worm / @skeleton_indices / skeleton_indices.m

        """

        # parition_type is assumed to be a key for the dictionary
        # worm_partition_subsets
        p = self.worm_partition_subsets[partition_type]

        # return only the subset of partitions contained in the particular
        # subset of interest, p.
        return {k: self.worm_partitions[k] for k in p}


    def get_subset_partition_mask(self, name):
        """
        Returns a boolean mask - for working with arrays given a partition.
        
        """
        keys = self.worm_partition_subsets[name]
        mask = np.zeros(49, dtype=bool)
        for key in keys:
            mask = mask | self.partition_mask(key)

        return mask


    def partition_mask(self, partition_key):
        """
        Returns a boolean numpy array corresponding to the partition requested.

        """
        mask = np.zeros(49, dtype=bool)
        slice_val = self.worm_partitions[partition_key]
        mask[slice(*slice_val)] = True
        return mask


    def get_partition(self, partition_key, data_key='skeletons',
                      split_spatial_dimensions=False):
        """    
        Retrieve partition of a measurement of the worm, that is, across all
        available frames but across only a subset of the 49 points.

        Parameters
        ---------------------------------------    
        partition_key: string
          The desired partition.  e.g. 'head', 'tail', etc.

          #TODO: This should be documented better 

          INPUT: a partition key, and an optional data key.
            If split_spatial_dimensions is True, the partition is returned 
            separated into x and y
          OUTPUT: a numpy array containing the data requested, cropped to just
                  the partition requested.
                  (so the shape might be, say, 4xn if data is 'angles')

        data_key: string  (optional)
          The desired measurement (default is 'skeletons')

        split_spatial_dimensions: bool    (optional)
          If True, the partition is returned separated into x and y

        Returns
        ---------------------------------------    
        A numpy array containing the data requested, cropped to just
        the partition requested.
        (so the shape might be, say, 4xn if data is 'angles')

        Notes
        ---------------------------------------    
        Translated from get.ALL_NORMAL_INDICES in SegwormMatlabClasses / 
        +seg_worm / @skeleton_indices / skeleton_indices.m

        """
        # We use numpy.split to split a data_dict element into three, cleaved
        # first by the first entry in the duple worm_partitions[partition_key],
        # and second by the second entry in that duple.

        # Taking the second element of the resulting list of arrays, i.e. [1],
        # gives the partitioned component we were looking for.
        partition = np.split(getattr(self,data_key),
                             self.worm_partitions[partition_key])[1]

        if(split_spatial_dimensions):
            return partition[:, 0, :], partition[:, 1,:]
        else:
            return partition


    def rotate(self, theta_d):
        """   
        Returns a NormalizedWorm instance with each frame rotated by 
        the amount given in the per-frame theta_d array.

        Parameters
        ---------------------------------------    
        theta_d: 1-dimensional ndarray of dtype=float
          The frame-by-frame rotation angle in degrees.
          A 1-dimensional n-element array where n is the number of
          frames, giving a rotation angle for each frame.

        Returns
        ---------------------------------------    
        A new NormalizedWorm instance with the same worm, rotated
        in each frame by the requested amount.

        """
        #theta_r = theta_d * (np.pi / 180)

        #%Unrotate worm
        #%-----------------------------------------------------------------
        # wwx = bsxfun(@times,sx,cos(theta_r)) + bsxfun(@times,sy,sin(theta_r));
        # wwy = bsxfun(@times,sx,-sin(theta_r)) +
        # bsxfun(@times,sy,cos(theta_r));

        # TODO
        return self


    @property
    def centre(self):
        """
        Frame-by-frame mean of the skeleton points

        Returns
        ---------------------------------------    
        A numpy array of length n, where n is the number of
        frames, giving for each frame the mean of the skeleton points.

        """
        s = self.skeletons
        with warnings.catch_warnings():
            temp = np.nanmean(s, 0, keepdims=False)

        return temp


    @property
    def angle(self):
        """
        Frame-by-frame mean of the skeleton points

        Returns
        ---------------------------------------    
        A numpy array of length n, giving for each frame
        the angle formed by the first and last skeleton point.

        """
        s = self.skeletons
        # obtain vector between first and last skeleton point
        v = s[48, :,:]-s[0,:,:]  
        # find the angle of this vector
        return np.arctan(v[1, :]/v[0,:])*(180/np.pi)


    def translate_to_centre(self):
        """ 
        Return a NormalizedWorm instance with each frame moved so the 
        centroid of the worm is 0,0

        Returns
        ---------------------------------------    
        A NormalizedWorm instance with the above properties.

        """
        s = self.skeletons
        s_mean = np.ones(np.shape(s)) * np.nanmean(s, 0, keepdims=False)

        #nw2 = NormalizedWorm()

        # TODO
        return s - s_mean


    def rotate_and_translate(self):
        """
        Perform both a rotation and a translation of the skeleton

        Returns
        ---------------------------------------    
        A numpy array, which is the centred and rotated normalized
        worm skeleton.

        Notes
        ---------------------------------------    
        To perform this matrix multiplication we are multiplying:
          rot_matrix * s
        This is shape 2 x 2 x n, times 2 x 49 x n.
        Basically we want the first matrix treated as two-dimensional,
        and the second matrix treated as one-dimensional,
        with the results applied elementwise in the other dimensions.

        To make this work I believe we need to pre-broadcast rot_matrix into
        the skeleton points dimension (the one with 49 points) so that we have
          2 x 2 x 49 x n, times 2 x 49 x n
        #s1 = np.rollaxis(self.skeletons, 1)

        #rot_matrix = np.ones(np.shape(s1)) * rot_matrix

        #self.skeletons_rotated = rot_matrix.dot(self.skeletons)    

        """

        skeletons_centred = self.translate_to_centre()
        orientation = self.angle

        a = -orientation * (np.pi / 180)

        rot_matrix = np.array([[np.cos(a), -np.sin(a)],
                               [np.sin(a),  np.cos(a)]])

        # we need the x,y listed in the first dimension
        s1 = np.rollaxis(skeletons_centred, 1)

        # for example, here is the first point of the first frame rotated:
        # rot_matrix[:,:,0].dot(s1[:,0,0])

        # ATTEMPTING TO CHANGE rot_matrix from 2x2x49xn to 2x49xn
        # rot_matrix2 = np.ones((2, 2, np.shape(s1)[1], np.shape(s1)[2])) * rot_matrix

        s1_rotated = []

        # rotate the worm frame-by-frame and add these skeletons to a list
        for frame_index in range(self.num_frames):
            s1_rotated.append(rot_matrix[:, :, frame_index].dot(s1[:,:, frame_index]))
        # print(np.shape(np.rollaxis(rot_matrix[:,:,0].dot(s1[:,:,0]),0)))

        # save the list as a numpy array
        s1_rotated = np.array(s1_rotated)

        # fix the axis settings
        return np.rollaxis(np.rollaxis(s1_rotated, 0, 3), 1)


    @property
    def num_frames(self):
        """ 
        The number of frames in the video.

        Returns
        ---------------------------------------    
        int
          number of frames in the video

        """

        # ndarray.shape returns a tuple of array dimensions.
        # the frames are along the first dimension i.e. [0].
        return self.skeletons.shape[2]

    @property
    def is_segmented(self):
        """
        Returns a 1-d boolean numpy array of whether 
        or not, frame-by-frame, the given frame was segmented

        """
        return self.segmentation_status == 's'

    def position_limits(self, dimension, measurement='skeletons'):
        """ 
        Maximum extent of worm's travels projected onto a given axis

        Parameters    
        ---------------------------------------        
        dimension: specify 0 for X axis, or 1 for Y axis.

        Notes
        ---------------------------------------    
        Dropped frames show up as NaN.
        nanmin returns the min ignoring such NaNs.

        """
        d = getattr(self,measurement)
        if(len(np.shape(d)) < 3):
            raise Exception("Position Limits Is Only Implemented for 2D data")
        return (np.nanmin(d[dimension, 0, :]), 
                np.nanmax(d[dimension, 1, :]))

    @property
    def contour_x(self):
        """ 
          Return the approximate worm contour, derived from data
          NOTE: The first and last points are duplicates, so we omit
                those on the second set. We also reverse the contour so that
                it encompasses an "out and back" contour
        """
        vc = self.vulva_contours
        nvc = self.non_vulva_contours
        return np.concatenate((vc[:, 0, :], nvc[-2:0:-1, 0,:]))    

    @property
    def contour_y(self):
        vc = self.vulva_contours
        nvc = self.non_vulva_contours
        return np.concatenate((vc[:, 1, :], nvc[-2:0:-1, 1,:]))    

    @property
    def skeleton_x(self):
        return self.skeletons[:, 0, :]

    @property
    def skeleton_y(self):
        return self.skeletons[:, 1, :]

    def __repr__(self):
        #TODO: This omits the properties above ...
        return utils.print_object(self)

class WormParsing(object):

    """
    This might eventually move somewhere else, but at least it is contained within
    the class. It was originally in the Normalized Worm code which was making things
    a bit overwhelming.
    
    TODO: Self does not refer to WormParsing ...
    
    """

    @staticmethod
    def h__computeNormalVectors(data):
        dx = np.gradient(data[:,0])
        dy = np.gradient(data[:,1])
        
        #This approach gives us -1 for the projection
        #We could also use:
        #dx_norm = -dy;
        #dy_norm = dx;
        #
        #and we would get 1 for the projection
        dx_norm = dy;
        dy_norm = -dx;
        
        vc_d_magnitude = np.sqrt(dx_norm**2 + dy_norm**2);
        
        norm_x = dx_norm/vc_d_magnitude;
        norm_y = dy_norm/vc_d_magnitude;
        
        return norm_x,norm_y

    @staticmethod
    def h__roundToOdd(value):
        value = np.floor(value)
        if value % 2 == 0:
            value = value + 1
            
        return value

    @staticmethod
    def h__getBounds(n1,n2,p_left,p_right):
        """
        
        Returns slice starts and stops
        #TODO: Rename everything to start and stop
        """
        pct = np.linspace(0,1,n1);
        left_pct = pct - p_left;
        right_pct = pct + p_right;

        left_I = np.floor(left_pct*n2);
        right_I = np.ceil(right_pct*n2);
        left_I[left_I < 0] = 0;
        right_I[right_I >= n2] = n2-1;
        right_I += 1
        return left_I,right_I

    @staticmethod
    def computeWidths(nw,vulva_contours,non_vulva_contours):
        """
        
        """        

        #Widths:
        #------------------------------------
        #The caller:
        #https://github.com/JimHokanson/SegwormMatlabClasses/blob/master/%2Bseg_worm/%2Bworm/%40skeleton/linearSkeleton.m
        #see helper__skeletonize - callls seg_worm.cv.skeletonize
        #
        #
        #Initial skeletonization:
        #https://github.com/JimHokanson/SegwormMatlabClasses/blob/master/%2Bseg_worm/%2Bcv/skeletonize.m
        #
        #Some refinement:
        #https://github.com/JimHokanson/SegwormMatlabClasses/blob/master/%2Bseg_worm/%2Bworm/%40skeleton/cleanSkeleton.m        
        
        #Widths are simply the distance between two "corresponding" sides of
        #the contour. The question is how to get these two locations. 

        FRACTION_WORM_SMOOTH = 1.0/12.0
        SMOOTHING_ORDER = 3
        PERCENT_BACK_SEARCH = 0.3;
        PERCENT_FORWARD_SEARCH = 0.3;
        END_S1_WALK_PCT = 0.15;
        

        n_frames = len(vulva_contours)
        data = np.full([nw.N_POINTS_NORMALIZED,n_frames],np.NaN)

        for iFrame, (s1,s2) in enumerate(zip(vulva_contours,non_vulva_contours)):
            
            # * I'm writing the code based on awesome_contours_oh_yeah_v2
            #   in Jim's testing folder            
            
            #Step 1: filter
            filter_width_s1 = WormParsing.h__roundToOdd(s1.shape[1]*FRACTION_WORM_SMOOTH)    
            s1[0,:] = sgolay(s1[0,:],filter_width_s1,SMOOTHING_ORDER)
            s1[1,:] = sgolay(s1[1,:],filter_width_s1,SMOOTHING_ORDER)

            filter_width_s2 = WormParsing.h__roundToOdd(s2.shape[1]*FRACTION_WORM_SMOOTH)    
            s2[0,:] = sgolay(s2[0,:],filter_width_s2,SMOOTHING_ORDER)
            s2[1,:] = sgolay(s2[1,:],filter_width_s2,SMOOTHING_ORDER)

            #TODO: Allow downsampling if the # of points is rediculous
            #200 points seems to be a good #
            #This operation gives us a matrix that is len(s1) x len(s2)
            dx_across = np.transpose(s1[0:1,:]) - s2[0,:]
            dy_across = np.transpose(s1[1:2,:]) - s2[1,:]
            d_across = np.sqrt(dx_across**2 + dy_across**2)
            dx_across = dx_across/d_across
            dy_across = dy_across/d_across
            
            #All s1 matching to s2
            #---------------------------------------
            left_I,right_I = WormParsing.h__getBounds(s1.shape[1],s2.shape[1],PERCENT_BACK_SEARCH,PERCENT_FORWARD_SEARCH)
            
            #%For each point on side 1, calculate normalized orthogonal values
            norm_x,norm_y = WormParsing.h__computeNormalVectors(s1)

            import pdb
            pdb.set_trace()
            
            #JAH: At this point, below is not implemented            
            
            #%For each point on side 1, find which side 2 the point pairs with
            dp_values1,match_I1 = WormParsing.h__getMatches(s1,s2,norm_x,norm_y,dx_across,dy_across,d_across,left_I,right_I)



               
        

          
                

#                import matplotlib.pyplot as plt
#                plt.scatter(vc[0,:],vc[1,:])
#                plt.scatter(nvc[0,:],nvc[1,:])
#                plt.gca().set_aspect('equal', adjustable='box')
#                plt.show()
#                
#                plt.plot(x_plot,y_plot)
#                plt.show()
            
            """
            #TODO: Make sure to bound
            cur_output_I = 0
            
            vc_I  = int(1)
            nvc_I = int(1)
            n_points = vc.shape[1]
            contour_widths = np.zeros(n_points*2)
            
            cur_xy_I = -3
            x_plot = np.full(n_points*5,np.NaN)
            y_plot = np.full(n_points*5,np.NaN)
            x_plot2 = np.full(n_points*5,np.NaN)
            y_plot2 = np.full(n_points*5,np.NaN)
            x_plot3 = np.full(n_points*5,np.NaN)
            y_plot3 = np.full(n_points*5,np.NaN)
            
            while (nvc_I != (n_points-2)) and (vc_I != (n_points-2)):
                cur_xy_I += 3 #skip a NaN too
                cur_output_I += 1             
             
                next_vc_I  = vc_I + 1
                next_nvc_I = nvc_I + 1
                
                if next_vc_I == 215:
                    pdb.set_trace()
                v_vc   = vc[:,next_vc_I] - vc[:,vc_I]     #dnj1
                v_nvc  = nvc[:,next_nvc_I] - nvc[:,nvc_I] #dnj2
                d_next = np.sum((vc[:,next_vc_I]-nvc[:,next_nvc_I])**2) #d12
                d_vc   = np.sum((vc[:,next_vc_I]-nvc[:,nvc_I])**2) #d1
                d_nvc  = np.sum((vc[:,next_nvc_I]-nvc[:,vc_I])**2) #d2 - nvc
              
                #(d_vc == d_nvc) or 
                if ((d_next <= d_vc) and (d_next <= d_nvc)):
                    vc_I = next_vc_I
                    nvc_I = next_nvc_I
                    contour_widths[cur_output_I] = np.sqrt(d_next)
                    
                    x_plot[cur_xy_I] = vc[0,vc_I]
                    y_plot[cur_xy_I] = vc[1,vc_I]   
                    x_plot[cur_xy_I+1] = nvc[0,nvc_I]
                    y_plot[cur_xy_I+1] = nvc[1,nvc_I]
                    
                    
                elif np.all((v_vc*v_nvc) >= 0):
                #contours go in similar directions
                #
                #Multiplication is checking that we have +*+ or -*- or 
                #a zero or two thrown in there (for the x & ys)
                #
                #NOTE: in general we want the smallest width as this is indicative
                #of being orthogonal to the direction of the body where
                #as going across the body at an angle will increase the apparent 
                #width (indicating that the larger width is not appropriate)
                
                    if d_vc < d_nvc:
                        vc_I = next_vc_I
                        contour_widths[cur_output_I] = np.sqrt(d_vc)
                        
                        x_plot2[cur_xy_I] = vc[0,next_vc_I]
                        y_plot2[cur_xy_I] = vc[1,next_vc_I]   
                        x_plot2[cur_xy_I+1] = nvc[0,nvc_I]
                        y_plot2[cur_xy_I+1] = nvc[1,nvc_I]                        
                        
                        
                    else:
                        nvc_I = next_nvc_I
                        contour_widths[cur_output_I] = np.sqrt(d_nvc)

                        x_plot3[cur_xy_I] = vc[0,next_vc_I]
                        y_plot3[cur_xy_I] = vc[1,next_vc_I]   
                        x_plot3[cur_xy_I+1] = nvc[0,nvc_I]
                        y_plot3[cur_xy_I+1] = nvc[1,nvc_I] 
                        
                else:        
                # The contours go in opposite directions.
                # Follow decreasing widths or walk along both contours.
                # In other words, catch up both contours, then walk along both.
                # Note: this step negotiates hairpin turns and bulges.                        
                    prev_width = contour_widths[cur_output_I]**2
                    if ((d_next <= d_vc) and (d_next <= d_nvc)) or ((d_vc > prev_width) and (d_nvc > prev_width)):
                        vc_I = next_vc_I
                        nvc_I = next_nvc_I
                        contour_widths[cur_output_I] = np.sqrt(d_next)  
                    elif d_vc < d_nvc:
                        vc_I = next_vc_I
                        contour_widths[cur_output_I] = np.sqrt(d_vc)
                    else:
                        nvc_I = next_nvc_I
                        contour_widths[cur_output_I] = np.sqrt(d_nvc)
                        
                    
                temp_widths_list.append(contour_widths)    
            """



    @staticmethod
    def computeSkeletonLengths(nw,xy_all):
        """

        Computes the running length (cumulative distance from start - head?) 
        for each skeleton.
        
        
        Parameters
        ----------
        xy_all : [numpy.array]
            Contains the skeleton positions for each frame.
            List length: # of frames
            Each element contains a numpy array of size [n_points x 2]
            Skeleton 
        """
        n_frames = len(xy_all)
        data = np.full([nw.N_POINTS_NORMALIZED,n_frames],np.NaN)
        for iFrame, cur_xy in enumerate(xy_all):
            if len(cur_xy) is not 0:
                sx = cur_xy[0,:]
                sy = cur_xy[1,:]
                cc = WormParsing.computeChainCodeLengths(sx,sy)
                data[:,iFrame] = WormParsing.normalizeParameter(nw,cc,cc)
                
        return data

    @staticmethod
    def computeChainCodeLengths(x,y):
        """
        Calculate the distance between a set of points and then calculate
        their cumulative distance from the first point.
        
        The first value returned has a value of 0 by definition.
        """
        
        #TODO: Should handle empty set - remove adding 0 as first element        
        
        #TODO: We need this for lengths as well, but the matrix vs vector 
        #complicates things
        
        dx = np.diff(x)
        dy = np.diff(y)
        
        distances = np.concatenate([np.array([0.0]), np.sqrt(dx**2 + dy**2)])
        return np.cumsum(distances)

    @staticmethod
    def normalizeAllFramesXY(nw,prop_to_normalize):
            
        n_frames = len(prop_to_normalize)
        norm_data = np.full([nw.N_POINTS_NORMALIZED,2,n_frames],np.NaN)
        for iFrame, cur_frame_value in enumerate(prop_to_normalize):
            if len(cur_frame_value) is not 0:
                sx = cur_frame_value[0,:]
                sy = cur_frame_value[1,:]
                cc = WormParsing.computeChainCodeLengths(sx,sy)
                norm_data[:,0,iFrame] = WormParsing.normalizeParameter(nw,sx,cc)
                norm_data[:,1,iFrame] = WormParsing.normalizeParameter(nw,sy,cc)
        
        return norm_data            
    
    @staticmethod
    def normalizeAllFrames(nw,prop_to_normalize,xy_data):
            
        n_frames = len(prop_to_normalize)
        norm_data = np.full([self.N_POINTS_NORMALIZED,n_frames],np.NaN)
        for iFrame, (cur_frame_value,cur_xy) in enumerate(zip(prop_to_normalize,xy_data)):
            if len(cur_frame_value) is not 0:
                sx = cur_xy[0,:]
                sy = cur_xy[1,:]
                cc = WormParsing.computeChainCodeLengths(sx,sy)
                norm_data[:,iFrame] = WormParsing.normalizeParameter(nw,cur_frame_value,cc)
        
        return norm_data 

    @staticmethod
    def calculateAngles(self,skeletons):
    
        """
        #Angles
        #----------------------------------
        #https://github.com/JimHokanson/SegwormMatlabClasses/blob/master/%2Bseg_worm/%2Bworm/%40skeleton/skeleton.m
        #https://github.com/JimHokanson/SegwormMatlabClasses/tree/master/%2Bseg_worm/%2Bcv/curvature.m
        #
        #   Note, the above code is written for the non-normalized worm ...
        #   edge_length= total_length/12
        #
        #   Importantly, the above approach calculates angles not between
        #   neighboring pairs but over a longer stretch of pairs (pairs that
        #   exceed the edge length). The net effect of this approach is to
        #   smooth the angles
        
        #vertex index - first one where the distance from the tip to this point
        #is greater than the edge length
        
        
        #s = norm_data[]
        
        
        #temp_s = np.full([self.N_POINTS_NORMALIZED,n_frames],np.NaN)
        #for iFrame in range(n_frames):
        #   temp_   
        """                  

        temp_angle_list = []
                      
        for iFrame, cur_frame_value in enumerate(skeletons):
            if len(cur_frame_value) is 0:
                temp_angle_list.append([])
            else:
                sx = cur_frame_value[0,:]
                sy = cur_frame_value[1,:]
                cc = self._computeChainCodeLengths(sx,sy)
    
                #This is from the old code
                edge_length = cc[-1]/12               
                
                #We want all vertices to be defined, and if we look starting
                #at the left_I for a vertex, rather than vertex for left and right
                #then we could miss all middle points on worms being vertices
                
                left_lengths = cc - edge_length
                right_lengths = cc + edge_length
    
                valid_vertices_I = utils.find((left_lengths > cc[0]) & (right_lengths < cc[-1]))
                
                left_lengths = left_lengths[valid_vertices_I]
                right_lengths = right_lengths[valid_vertices_I]                
                
                left_x = np.interp(left_lengths,cc,sx)
                left_y = np.interp(left_lengths,cc,sy)
            
                right_x = np.interp(right_lengths,cc,sx)
                right_y = np.interp(right_lengths,cc,sy)
    
                d2_y = sy[valid_vertices_I] - right_y
                d2_x = sx[valid_vertices_I] - right_x
                d1_y = left_y - sy[valid_vertices_I]
                d1_x = left_x - sx[valid_vertices_I] 
    
                frame_angles = np.arctan2(d2_y,d2_x) - np.arctan2(d1_y,d1_x)
                
                frame_angles[frame_angles > np.pi] -= 2*np.pi
                frame_angles[frame_angles < -np.pi] += 2*np.pi
                
                frame_angles *= 180/np.pi
                
                all_frame_angles = np.full_like(cc,np.NaN)
                all_frame_angles[valid_vertices_I] = frame_angles
                
                temp_angle_list.append(all_frame_angles)
                
        return WormParsing.normalizeAllFrames(self,temp_angle_list,skeletons)
    
    @staticmethod
    def normalizeParameter(self,orig_data,old_lengths):
        """
        
        This function finds where all of the new points will be when evenly
        sampled (in terms of chain code length) from the first to the last 
        point in the old data.

        These points are then related to the old points. If a new points is at
        an old point, the old point data value is used. If it is between two
        old points, then linear interpolation is used to determine the new value
        based on the neighboring old values.

        NOTE: For better or worse, this approach does not smooth the new data
        
        Old Code:
        https://github.com/openworm/SegWorm/blob/master/ComputerVision/chainCodeLengthInterp.m  
        
        Parameters:
        -----------
        non_normalizied_data :
            - ()
        """
        
        
        #TODO: Might just replace all of this with an interpolation call
        
        new_lengths = np.linspace(old_lengths[0],old_lengths[-1],self.N_POINTS_NORMALIZED)
        
        #For each point, get the bordering points
        #Sort, with old coming before new
        I = np.argsort(np.concatenate([old_lengths, new_lengths]), kind='mergesort')
        #Find new points, an old point will be to the left
        new_I = utils.find(I >= len(old_lengths)) #indices 0 to n-1, look for >= not >
        
        norm_data = np.empty_like(new_lengths)        
        
        #Can we do this without a loop (YES!)
        #find those that are equal
        #those that are not equal (at an old point) then do vector math        

        for iSeg,cur_new_I in enumerate(new_I):
            cur_left_I = I[cur_new_I-1]
            cur_right_I = cur_left_I + 1
            if iSeg == 0 or (iSeg == len(new_lengths) - 1) or (new_lengths[iSeg] == old_lengths[cur_left_I]):
                norm_data[iSeg] = orig_data[cur_left_I]
            else:
                new_position = new_lengths[iSeg]
                left_position = old_lengths[cur_left_I]
                right_position = old_lengths[cur_right_I]                    
                total_length = right_position - left_position
                #NOTE: If we are really close to left, then we want mostly
                #left, which means right_position - new_position will almost
                #be equal to the total length, and left_pct will be close to 1
                left_pct = (right_position - new_position)/total_length
                right_pct = (new_position - left_position)/total_length
                norm_data[iSeg] = left_pct*orig_data[cur_left_I] + right_pct*orig_data[cur_right_I]


        return norm_data        

class SkeletonPartitions(object):
    
    """
    The idea with this class was to move details regarding how the worm can be
    divided up 
    """
    #TODO: This needs to be implemented
    pass