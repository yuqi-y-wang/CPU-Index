import os
import numpy as np
import pandas as pd
from skimage import io as skio
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
print(currentdir, parentdir)

import MODULES
from MODULES import preprocess_dict
from MODULES.preprocess_image_funcs import process_obj

if __name__ == "__main__":
    df = pd.read_csv(preprocess_dict.df_dir,index_col=0)
    df.reset_index(inplace=True)
    df['num_lesions'] = None
    df['min_lesion_size'] = None
    df['max_lesion_size'] = None
    df['mean_lesion_size'] = None
    df['std_lesion_size'] = None
    for i in range(preprocess_dict.num_parts*3):
        df[f'lesion_locations_{i}'] = None
        
    for i, row in df.iterrows():
        name_annotation, name_img = row['annotation_name'], row['img_name']
        annotation_path = f'{preprocess_dict.annotation_dir}{name_annotation}'
        nifti_path = f'{preprocess_dict.img_dir}{name_img}/{name_img}.nii.gz'
        n_lesions, sizes, location, mask = process_obj(nifti_path, annotation_path)
        df.loc[i, 'num_lesions'] = n_lesions
        df.loc[i, 'max_lesion_size'] = max(sizes)
        df.loc[i, 'min_lesion_size'] = np.min(sizes)
        df.loc[i, 'mean_lesion_size'] = np.mean(sizes)
        df.loc[i, 'std_lesion_size'] = np.std(sizes)
        for j in range(preprocess_dict.num_parts*3):
            df.loc[i, f'lesion_locations_{j}'] = location[j]
        ### if save
        mask = mask.astype('u1')
        np.save(f'{preprocess_dict.mask_dir}{name_img}.npy', mask)
    df.set_index('img_name', inplace=True)
    df.to_csv(preprocess_dict.df_dir)