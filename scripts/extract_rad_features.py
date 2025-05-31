import os
import numpy as np
import pandas as pd
from skimage import io as skio

import MODULES
from MODULES import preprocess_dict
from MODULES.preprocess_image_funcs import process_obj
from tqdm import tqdm

import ast
import radiomics
from radiomics import featureextractor
from skimage.measure import regionprops, label
import SimpleITK as sitk
import openpyxl
import nibabel as nib

if __name__ == "__main__":  
    df = pd.read_csv(preprocess_dict.df_dir,index_col=0)

    settings = {}
    settings['force2D'] = True
    settings['binWidth'] = 2
    # Instantiate the extractor
    extractor = featureextractor.RadiomicsFeatureExtractor(
        **settings)  # ** 'unpacks' the dictionary in the function call
    # enable 3D shape
    extractor.enableFeatureClassByName('shape')
    extractor.enableImageTypeByName('Wavelet')

    for i, row in tqdm(df.iterrows()):
        name_annotation, name_img = row['annotation_name'], row['img_name']
        nifti_path = f'{preprocess_dict.img_dir}{name_img}/{name_img}.nii.gz'
        mask_path = f'{preprocess_dict.mask_dir}/{name_img}.npy'

        img_array = nib.load(nifti_path).get_fdata()
        img_array = (img_array - np.min(img_array))/(np.max(img_array) - np.min(img_array))
        mask_array = np.load(mask_path)

        image = sitk.GetImageFromArray(img_array*255)
        mask = sitk.GetImageFromArray(mask_array*255)
        result = extractor.execute(image, mask, label=255)
        lists = result.items()
        x, y = zip(*lists)
        if i == 0:
            cols = list(x) 
            df_radfeatures = pd.DataFrame(columns=cols)
        row = list(y)
        df_radfeatures.loc[i] = row
    
    df_radfeatures.to_csv('rad_features.csv')