# -*- coding: utf-8 -*-
"""
morphology_features.py

"""
import numpy as np

from .. import utils

<<<<<<< HEAD
from . import feature_comparisons as fc

from matplotlib import pyplot as plt

=======
>>>>>>> 9435613fb3e6514ccef27c237f79fcf1a3ac8deb
class Widths(object):
    """
    Attributes
    ----------    
    head :
    midbody :
    tail :
    """
    
    fields = ('head', 'midbody', 'tail')
    
<<<<<<< HEAD
    def __init__(self,features_ref, explain=[]):
=======
    def __init__(self, features_ref):
>>>>>>> 9435613fb3e6514ccef27c237f79fcf1a3ac8deb
        """
        Parameters
        ----------
        features_ref : WormFeatures instance
        
        Note the current approach just computes the mean of the 
        different body section widths. Eventually this should be 
        computed in this class.

        """
        nw = features_ref.nw
    
        for partition in self.fields:
<<<<<<< HEAD
            setattr(self,partition, np.mean(nw.get_partition(partition, 'widths'),0))

        if 'width' in explain:
            self.explain(nw)

    def explain(self, nw):
        text1 = '''The worm segments are grouped into head, midbody, and tail. The plot below shows the widths for each section
        color coded by section.'''

        head = nw.get_partition('head','widths')
        mid = nw.get_partition('midbody','widths')
        tail = nw.get_partition('tail','widths')





        text2 = ''''''


=======
            widths_in_partition = nw.get_partition(partition, 'widths')
            setattr(self, partition, np.mean(widths_in_partition, 0))
>>>>>>> 9435613fb3e6514ccef27c237f79fcf1a3ac8deb
    
    @classmethod
    def from_disk(cls,width_ref):

        self = cls.__new__(cls)

        for partition in self.fields:
            widths_in_partition = utils._extract_time_from_disk(width_ref, 
                                                                partition)
            setattr(self, partition, widths_in_partition)
    
        return self
        
    def __eq__(self, other):
        return (
                utils.correlation(self.head, other.head, 
                                   'morph.width.head') and
                utils.correlation(self.midbody, other.midbody,
                                   'morph.width.midbody') and
                utils.correlation(self.tail, other.tail, 
                                   'morph.width.tail'))

    def __repr__(self):
        return utils.print_object(self)  
