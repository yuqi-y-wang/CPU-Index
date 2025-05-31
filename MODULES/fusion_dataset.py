import numpy as np
import torch.utils.data as data
import nibabel as nib

from torchvision import transforms
from MODULES.pe_functions import position_encoding
from scipy import ndimage

class FusionDataSet(data.Dataset):
    def __init__(self, ids, imgdirtemplate, maskdirtemplate, 
                 df, feature_cols, dim_pe=1024, 
                 bins=100,
                 input_D=100, input_H=200, input_W=200,
                 seg='normal', mode = 'Train'):
        '''
        ids: list of case ids
        imgdirtemplate: a string template for img with imgdirtemplate.format(id)
        maskdirtemplate: a string template for mask with maskdirtemplate.format(id)
        df: the dataFrame for non-img data
        feature_cols: feature cols
        seg: None: no application of mask;
            'normal': simple application of mask; 
            'gradient': gradient application of mask.
        '''
        self.ids = ids
        self.df = df
        self.feature_cols = feature_cols
        self.imagedirtemplate = imgdirtemplate
        self.maskdirtemplate = maskdirtemplate
        self.dim_pe = dim_pe
        self.seg = seg
        self.input_D = input_D
        self.input_H = input_H
        self.input_W = input_W
        self.mode = mode
        self.bins = bins
        self.times = [0., ]
        self.time_buckets = None
        self.num_times = 0
        self.get_times()
        
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        idx = self.ids[idx]
        image = nib.load(self.imagedirtemplate.replace('id', idx)).get_fdata()
        image = (image - np.min(image))/(np.max(image) - np.min(image))*255
        mask = np.load(self.maskdirtemplate.replace('id', idx))
        img = image * mask
        img = img.transpose(2 ,0, 1)
        img = self.resize_img(img)
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        img = np.nan_to_num(img)
        nonimg = self.df[self.feature_cols].loc[idx].values
        if self.dim_pe:
            if nonimg.ndim != 2:
                nonimg = nonimg.reshape(1, -1)
            nonimg = position_encoding(nonimg, D=self.dim_pe)
            nonimg = nonimg.flatten()
            nonimg = np.pad(nonimg, (self.dim_pe - len(nonimg), 0), 'edge')
        T = self.df['time'].loc[idx]
        E = self.df['event'].loc[idx]
        label = self.compute_multitask_label(T, E)
        return img.astype(np.float32), nonimg.astype(np.float32), label, T, E, idx
    
    
    def get_times(self, is_min_time_zero = True, extra_pct_time = 0.1):
        """ Building the time axis (self.times) as well as the time intervals 
            ( all the [ t(k-1), t(k) ) in the time axis.
        """

        # Setting the min_time and max_time
        T = self.df['time']
        max_time = max(T)
        if is_min_time_zero :
            min_time = 0. 
        else:
            min_time = min(T)
        
        # Setting optional extra percentage time
        if 0. <= extra_pct_time <= 1.:
            p = extra_pct_time
        else:
            raise Exception("extra_pct_time has to be between [0, 1].") 

        # Building time points and time buckets
        self.times = np.linspace(min_time, max_time*(1. + p), self.bins)
        self.get_time_buckets()
        self.num_times = len(self.time_buckets)
    
    def get_time_buckets(self, extra_timepoint=False):
        """ Creating the time buckets based on the times axis such that
            for the k-th time bin is [ t(k-1), t(k) ] in the time axis.
        """
        
        # Checking if the time axis has already been created
        if self.times is None or len(self.times) <= 1:
            error = 'The time axis needs to be created before'
            error += ' using the method get_time_buckets.'
            raise AttributeError(error)
        
        # Creating the base time buckets
        time_buckets = []
        for i in range(len(self.times)-1):
            time_buckets.append((self.times[i], self.times[i+1]))
        
        # Adding an additional element if specified
        if extra_timepoint:
            time_buckets += [ (time_buckets[-1][1], time_buckets[-1][1]*1.01) ]
        self.time_buckets = time_buckets
        
    def compute_multitask_label(self, T, E, extra_pct_time = 0.1, is_min_time_zero=True):
        """ Given the survival_times, events and time_points vectors, 
            it returns a ndarray of the encodings for all units 
            such that:
                Y = [[0, 0, 1, 0, 0],  # unit experienced an event at t = 3
                     [0, 1, 0, 0, 0],  # unit experienced an event at t = 2
                     [0, 1, 1, 1, 1],] # unit was censored at t = 2
        """ 
        y = np.zeros(self.bins)
        min_abs_value = [abs(a_j_1-T) for (a_j_1, a_j) in self.time_buckets]
        index = np.argmin(min_abs_value)
        if E == 1:
                y[index] = 1.
        else:                
            y[(index):] = 1.
        return y 

    def resize_img(self, img):
        """
        Resize the data to the input size
        """ 
        [depth, height, width] = img.shape
        scale = [self.input_D*1.0/depth, self.input_H*1.0/height, self.input_W*1.0/width]  
        img = ndimage.interpolation.zoom(img, scale, order=0)

        return img